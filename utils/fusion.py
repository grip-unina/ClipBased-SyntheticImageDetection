'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
from scipy.special import logsumexp, logit, log_expit
softplusinv = lambda x: np.log(np.expm1(x))  # log(exp(x)-1)
softminusinv = lambda x: x - np.log(-np.expm1(x)) # info: https://jiafulow.github.io/blog/2019/07/11/softplus-and-softminus/

fusion_functions = {
    'mean_logit'   : lambda x, axis: np.mean(x, axis),
    'max_logit'    : lambda x, axis: np.max(x, axis),
    'median_logit' : lambda x, axis: np.median(x, axis),
    'lse_logit'    : lambda x, axis: logsumexp(x, axis),
    'mean_prob'    : lambda x, axis: softminusinv(logsumexp(log_expit(x), axis) - np.log(x.shape[axis])),
    'soft_or_prob' : lambda x, axis: -softminusinv(np.sum(log_expit(-x), axis)),
}

def apply_fusion(x, typ, axis):
    return fusion_functions[typ](x, axis)
