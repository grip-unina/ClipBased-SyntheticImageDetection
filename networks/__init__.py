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

def create_architecture(name_arch, pretrained=False, num_classes=1):
    if name_arch == "res50nodown":
        from .resnet_mod import resnet50

        if pretrained:
            model = resnet50(pretrained=True, stride0=1, dropout=0.5).change_output(num_classes)
        else:
            model = resnet50(num_classes=num_classes, stride0=1, dropout=0.5)
    elif name_arch == "res50":
        from .resnet_mod import resnet50

        if pretrained:
            model = resnet50(pretrained=True, stride0=2).change_output(num_classes)
        else:
            model = resnet50(num_classes=num_classes, stride0=2)
    elif name_arch.startswith('opencliplinear_'):
        from .openclipnet import OpenClipLinear
        model = OpenClipLinear(num_classes=num_classes, pretrain=name_arch[15:], normalize=True)
    elif name_arch.startswith('opencliplinearnext_'):
        from .openclipnet import OpenClipLinear
        model = OpenClipLinear(num_classes=num_classes, pretrain=name_arch[19:], normalize=True, next_to_last=True)
    else:
        assert False
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_weights(model, model_path):
    from torch import load
    dat = load(model_path, map_location='cpu')
    if 'model' in dat:
        if ('module._conv_stem.weight' in dat['model']) or \
           ('module.fc.fc1.weight' in dat['model']) or \
           ('module.fc.weight' in dat['model']):
            model.load_state_dict(
                {key[7:]: dat['model'][key] for key in dat['model']})
        else:
            model.load_state_dict(dat['model'])
    elif 'state_dict' in dat:
        model.load_state_dict(dat['state_dict'])
    elif 'net' in dat:
        model.load_state_dict(dat['net'])
    elif 'main.0.weight' in dat:
        model.load_state_dict(dat)
    elif '_fc.weight' in dat:
        model.load_state_dict(dat)
    elif 'conv1.weight' in dat:
        model.load_state_dict(dat)
    else:
        print(list(dat.keys()))
        assert False
    return model
