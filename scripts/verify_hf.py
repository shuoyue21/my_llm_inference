#!/usr/bin/env python3
"""Independent FP32 Llama/Qwen2/Qwen3 logits from original HF safetensors.
Correctness only; timings from this diagnostic are not benchmark results.
"""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--binary',required=True);p.add_argument('--model-dir',type=Path,required=True)
 p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--tokens',type=int,default=6)
 p.add_argument('--prompt');p.add_argument('--output',type=Path)
 a=p.parse_args();torch.set_num_threads(1)
 cfg=json.loads((a.model_dir/'config.json').read_text());family=cfg['model_type']
 prompt=a.prompt or ('The capital of France is' if family=='llama' else '<|im_start|>user\n1+2等于几？<|im_end|>\n<|im_start|>assistant\n'+('<think>\n\n</think>\n\n' if family=='qwen3' else ''))
 tokenizer=Tokenizer.from_file(str(a.model_dir/'tokenizer.json'))
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 with tempfile.TemporaryDirectory() as temp:
  dump=Path(temp)/'logits.f32'
  cmd=[a.binary,str(a.model_dir/'model.bin'),str(a.model_dir/'tokenizer.json'),'--device',a.device,'--prompt',prompt,'--max-new-tokens',str(a.tokens),'--json','--dump-logits',str(dump)]
  run=subprocess.run(cmd,capture_output=True,text=True,env=env,check=True);result=json.loads(run.stdout)
  wanted=tokenizer.encode(prompt,add_special_tokens=(family=='llama')).ids
  assert result['prompt_tokens']==wanted,('tokenizer',result['prompt_tokens'],wanted)
  tokens=result['prompt_tokens']+result['generated_tokens'][:-1]
  actual=np.fromfile(dump,dtype='<f4').reshape(-1,cfg['vocab_size']);assert len(actual)==len(tokens)
  state={}
  for file in sorted(a.model_dir.glob('*.safetensors')):state.update({k:v.float() for k,v in load_file(str(file),device='cpu').items()})
  d=cfg['hidden_size'];heads=cfg['num_attention_heads'];kvheads=cfg['num_key_value_heads'];hs=cfg.get('head_dim',d//heads);layers=cfg['num_hidden_layers']
  keys=torch.empty(layers,len(tokens),kvheads,hs);values=torch.empty_like(keys)
  frequencies=1/(float(cfg.get('rope_theta',10000))**(torch.arange(0,hs,2).float()/hs))
  scaling=cfg.get('rope_scaling')
  if scaling:
   assert scaling.get('rope_type',scaling.get('type'))=='llama3'
   wavelength=2*torch.pi/frequencies;low=scaling['original_max_position_embeddings']/scaling['low_freq_factor'];high=scaling['original_max_position_embeddings']/scaling['high_freq_factor']
   scaled=torch.where(wavelength>low,frequencies/scaling['factor'],frequencies)
   smooth=(scaling['original_max_position_embeddings']/wavelength-scaling['low_freq_factor'])/(scaling['high_freq_factor']-scaling['low_freq_factor'])
   mixed=(1-smooth)*scaled/scaling['factor']+smooth*scaled
   frequencies=torch.where((wavelength<=low)&(wavelength>=high),mixed,scaled)
  def norm(x,w):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+cfg['rms_norm_eps'])*w
  def rotate(x,pos):
   x=x.reshape(-1,hs);left,right=x.chunk(2,-1);c=(pos*frequencies).cos();s=(pos*frequencies).sin();return torch.cat((left*c-right*s,left*s+right*c),-1)
  maximum=0.;rms_error=0.
  for pos,token in enumerate(tokens):
   x=state['model.embed_tokens.weight'][token].clone()
   for layer in range(layers):
    prefix=f'model.layers.{layer}.';z=norm(x,state[prefix+'input_layernorm.weight'])
    def project(name):
     value=state[prefix+f'self_attn.{name}_proj.weight']@z
     bias=state.get(prefix+f'self_attn.{name}_proj.bias');return value if bias is None else value+bias
    q=project('q').reshape(heads,hs);k=project('k').reshape(kvheads,hs)
    if family=='qwen3':q=norm(q,state[prefix+'self_attn.q_norm.weight']);k=norm(k,state[prefix+'self_attn.k_norm.weight'])
    q=rotate(q,pos);keys[layer,pos]=rotate(k,pos);values[layer,pos]=project('v').reshape(kvheads,hs)
    k=keys[layer,:pos+1].repeat_interleave(heads//kvheads,dim=1);v=values[layer,:pos+1].repeat_interleave(heads//kvheads,dim=1)
    scores=torch.einsum('hd,thd->ht',q,k)/(hs**0.5)
    attention=torch.einsum('ht,thd->hd',scores.softmax(-1),v).flatten();x=x+state[prefix+'self_attn.o_proj.weight']@attention
    z=norm(x,state[prefix+'post_attention_layernorm.weight'])
    x=x+state[prefix+'mlp.down_proj.weight']@(torch.nn.functional.silu(state[prefix+'mlp.gate_proj.weight']@z)*(state[prefix+'mlp.up_proj.weight']@z))
   classifier=state['model.embed_tokens.weight'] if cfg.get('tie_word_embeddings',False) else state['lm_head.weight']
   expected=(classifier@norm(x,state['model.norm.weight'])).numpy()
   np.testing.assert_allclose(actual[pos],expected,atol=6e-4,rtol=6e-4,err_msg=f'{family} {a.device} pos={pos}')
   maximum=max(maximum,float(np.max(np.abs(actual[pos]-expected))));rms_error+=float(np.mean((actual[pos]-expected)**2))
   if pos>=len(wanted)-1:assert int(expected.argmax())==result['generated_tokens'][pos-len(wanted)+1],('greedy',pos)
  report={'family':family,'device':a.device,'model_dir':str(a.model_dir),'positions':len(tokens),'vocab_size':cfg['vocab_size'],'max_abs_logit_error':maximum,'rms_logit_error':(rms_error/len(tokens))**.5,'tokenizer_match':True,'generated_tokens_match':True,'text':result['text'],'source':json.loads((a.model_dir/'source.json').read_text()) if (a.model_dir/'source.json').exists() else None}
  if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
