#!/usr/bin/env python3
"""Validate Qwen2.5 tokenizer and logits against HF tokenizer.json + original safetensors.
Requires torch, safetensors, tokenizers, numpy. No transformers or network needed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--binary",default="build-qwen2/demo/llama_infer")
p.add_argument("--model-dir",type=Path,default=Path("models/qwen2.5-0.5b-instruct"))
p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
p.add_argument("--tokens",type=int,default=4)
a=p.parse_args()
torch.set_num_threads(1)
config=json.loads((a.model_dir/"config.json").read_text())
tokenizer=Tokenizer.from_file(str(a.model_dir/"tokenizer.json"))
prompt="<|im_start|>user\n你好，1+2等于几？<|im_end|>\n<|im_start|>assistant\n"
env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
with tempfile.TemporaryDirectory() as directory:
    dump=Path(directory)/"logits.f32"
    command=[a.binary,str(a.model_dir/"model.bin"),str(a.model_dir/"tokenizer.json"),
             "--device",a.device,"--prompt",prompt,"--max-new-tokens",str(a.tokens),"--json","--dump-logits",str(dump)]
    run=subprocess.run(command,check=True,capture_output=True,text=True,env=env)
    result=json.loads(run.stdout)
    assert result["prompt_tokens"]==tokenizer.encode(prompt,add_special_tokens=False).ids
    actual=np.fromfile(dump,dtype="<f4").reshape(-1,config["vocab_size"])
    # Include tricky whitespace, punctuation and literal byte-level alphabet characters.
    for text in ["Hello  world\n\n", "中文12345测试", "foo   bar\t baz", "Ġhello🙂", "line\n  "]:
        r=subprocess.run([a.binary,str(a.model_dir/"model.bin"),str(a.model_dir/"tokenizer.json"),
            "--device","cpu","--prompt",text,"--max-new-tokens","0","--json"],check=True,capture_output=True,text=True,env=env)
        assert json.loads(r.stdout)["prompt_tokens"]==tokenizer.encode(text,add_special_tokens=False).ids, repr(text)
    state={k:v.float() for k,v in load_file(str(a.model_dir/"model.safetensors"),device="cpu").items()}
    d=config["hidden_size"]; heads=config["num_attention_heads"]; kvheads=config["num_key_value_heads"]; hs=d//heads
    tokens=result["prompt_tokens"]+result["generated_tokens"][:-1]
    assert len(tokens)==len(actual)
    keys=torch.empty(config["num_hidden_layers"],len(tokens),kvheads,hs); values=torch.empty_like(keys)
    freq=1/(float(config["rope_theta"])**(torch.arange(0,hs,2).float()/hs))
    def norm(x,w): return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+config["rms_norm_eps"])*w
    def rotate(x,pos):
        x=x.reshape(-1,hs); left,right=x.chunk(2,-1); c=(pos*freq).cos(); s=(pos*freq).sin()
        return torch.cat((left*c-right*s,left*s+right*c),-1)
    max_error=0.0
    for pos,token in enumerate(tokens):
        x=state["model.embed_tokens.weight"][token].clone()
        for layer in range(config["num_hidden_layers"]):
            prefix=f"model.layers.{layer}."
            z=norm(x,state[prefix+"input_layernorm.weight"])
            def proj(name): return state[prefix+f"self_attn.{name}_proj.weight"]@z+state[prefix+f"self_attn.{name}_proj.bias"]
            q=rotate(proj("q"),pos); keys[layer,pos]=rotate(proj("k"),pos); values[layer,pos]=proj("v").reshape(kvheads,hs)
            k=keys[layer,:pos+1].repeat_interleave(heads//kvheads,dim=1)
            v=values[layer,:pos+1].repeat_interleave(heads//kvheads,dim=1)
            scores=torch.einsum("hd,thd->ht",q,k)/(hs**0.5)
            attention=torch.einsum("ht,thd->hd",scores.softmax(-1),v).flatten()
            x=x+state[prefix+"self_attn.o_proj.weight"]@attention
            z=norm(x,state[prefix+"post_attention_layernorm.weight"])
            x=x+state[prefix+"mlp.down_proj.weight"]@(torch.nn.functional.silu(state[prefix+"mlp.gate_proj.weight"]@z)*(state[prefix+"mlp.up_proj.weight"]@z))
        final=norm(x,state["model.norm.weight"])
        classifier=state["model.embed_tokens.weight"] if config["tie_word_embeddings"] else state["lm_head.weight"]
        expected=(classifier@final).numpy()
        np.testing.assert_allclose(actual[pos],expected,atol=5e-4,rtol=5e-4,err_msg=f"position={pos}")
        max_error=max(max_error,float(np.max(np.abs(actual[pos]-expected))))
        if pos>=len(result["prompt_tokens"])-1: assert int(expected.argmax())==result["generated_tokens"][pos-len(result["prompt_tokens"])+1]
    print(json.dumps({"device":a.device,"positions":len(tokens),"max_abs_logit_error":max_error,
                      "tokenizer_cases_match":6,"generated_tokens_match":True,"text":result["text"]},ensure_ascii=False))
