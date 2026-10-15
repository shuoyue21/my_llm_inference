#!/usr/bin/env python3
"""Export FP32 legacy checkpoints from original HF Llama/Qwen safetensors + runtime metadata."""
import argparse,json,struct
from pathlib import Path
import numpy as np

class Tensors:
    def __init__(self, folder):
        self.items={}; self.maps=[]
        for file in sorted(folder.glob('*.safetensors')):
            with file.open('rb') as f:
                length=struct.unpack('<Q',f.read(8))[0]; metadata=json.loads(f.read(length))
            data=np.memmap(file,mode='r',dtype=np.uint8,offset=8+length); self.maps.append(data)
            for name,info in metadata.items():
                if name!='__metadata__':self.items[name]=(data,info)
        if not self.items:raise ValueError('No safetensors found')
    def write(self,out,name):
        data,info=self.items[name]; begin,end=info['data_offsets']; data=data[begin:end]
        if info['dtype']=='BF16':array=(data.view('<u2').astype('<u4')<<16).view('<f4')
        elif info['dtype']=='F16':array=data.view('<f2').astype('<f4')
        elif info['dtype']=='F32':array=data.view('<f4')
        else:raise ValueError(info['dtype'])
        if array.size!=int(np.prod(info['shape'])):raise ValueError(name+' has invalid shape')
        array.tofile(out)

def export(folder,output,context):
    config=json.loads((folder/'config.json').read_text()); family=config['model_type']
    if family not in ['llama','qwen2','qwen3']:raise ValueError('Unsupported model type '+family)
    if not 0<context<=config['max_position_embeddings']:raise ValueError('Invalid context')
    if config.get('hidden_act','silu')!='silu':raise ValueError('Only SwiGLU is supported')
    t=Tensors(folder); d=config['hidden_size']; h=config['intermediate_size']; l=config['num_hidden_layers']
    heads=config['num_attention_heads']; kvheads=config['num_key_value_heads']; hd=config.get('head_dim',d//heads)
    shared=config.get('tie_word_embeddings',False); vocab=config['vocab_size']
    def layers(out,suffix):
        for layer in range(l):t.write(out,f'model.layers.{layer}.{suffix}')
    temporary=output.with_name(output.name+'.partial')
    with temporary.open('wb') as out:
        if family=='qwen3':
            out.write(struct.pack('<8i',heads*hd,d,l,heads,kvheads,vocab if shared else -vocab,context,h))
            layers(out,'input_layernorm.weight');layers(out,'post_attention_layernorm.weight');t.write(out,'model.norm.weight')
            t.write(out,'model.embed_tokens.weight')
            layers(out,'self_attn.q_proj.weight');layers(out,'self_attn.q_norm.weight')
            layers(out,'self_attn.k_proj.weight');layers(out,'self_attn.k_norm.weight')
            layers(out,'self_attn.v_proj.weight');layers(out,'self_attn.o_proj.weight')
            for projection in ['gate','down','up']:layers(out,f'mlp.{projection}_proj.weight')
            t.write(out,'model.embed_tokens.weight' if shared else 'lm_head.weight')
        else:
            if heads*hd!=d:raise ValueError('Separate head_dim currently requires Qwen3')
            out.write(struct.pack('<7i',d,h,l,heads,kvheads,vocab if shared else -vocab,context))
            t.write(out,'model.embed_tokens.weight');layers(out,'input_layernorm.weight')
            for proj in ['q','k','v']:
                for layer in range(l):
                    t.write(out,f'model.layers.{layer}.self_attn.{proj}_proj.weight')
                    if family=='qwen2':t.write(out,f'model.layers.{layer}.self_attn.{proj}_proj.bias')
            layers(out,'self_attn.o_proj.weight');layers(out,'post_attention_layernorm.weight')
            for projection in ['gate','down','up']:layers(out,f'mlp.{projection}_proj.weight')
            t.write(out,'model.norm.weight');np.zeros(context*hd,dtype='<f4').tofile(out)
            if not shared:t.write(out,'lm_head.weight')
    temporary.replace(output)
    metadata={'model_type':family,'hidden_size':d,'attention_dim':heads*hd,'head_dim':hd,
              'rope_theta':config.get('rope_theta',10000),'rope_scaling':config.get('rope_scaling'),
              'rms_norm_eps':config.get('rms_norm_eps',1e-5),'source_config':config}
    if (folder/'source.json').exists():metadata['source']=json.loads((folder/'source.json').read_text())
    Path(str(output)+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'output':str(output),'bytes':output.stat().st_size,'family':family,'context':context}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model_dir',type=Path);p.add_argument('output',type=Path);p.add_argument('--context',type=int,default=2048)
    a=p.parse_args();export(a.model_dir,a.output,a.context)
