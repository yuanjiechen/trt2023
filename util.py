import torch
import re


name_width = 50 # max width of layer names
qname_width = name_width + 20 # max width of quantizer names

class Logger:
    def info(self, s):
        print("INFO:", s)
    def warn(self, s):
        print("WARN:", s)
logger = Logger()


def set_quantizer(name, mod, quantizer,k, v):
    """Set attributes for mod.quantizer."""

    quantizer_mod = getattr(mod, quantizer, None)
    if quantizer_mod is not None:
        assert hasattr(quantizer_mod,k)
        setattr(quantizer_mod,k, v)
    else:
        logger.warn(f'{name} has no {quantizer}')

def set_quantizers(name, mod, which='both', **kwargs):  #key->_disable  value->True
    """Set quantizer attributes for mod."""

    s = f'Warning: changing {which} quantizers of {name:{qname_width}}'
    for k, v in kwargs.items():
        s += (f' {k}={v}')
        if which in ['input', 'both']:
            set_quantizer(name, mod, '_input_quantizer', k, v)
        if which in ['weight', 'both']:
            set_quantizer(name, mod, '_weight_quantizer', k, v)
    logger.info(s)

def set_quantizer_by_name(model, names, **kwargs):
    """Set quantizer attributes for layers where name contains a substring in names."""

    for name, mod in model.named_modules():
        for n in names:
            if re.search(n,name):
                for name_mod , mod_sub in mod.named_modules():
                    if hasattr(mod_sub, '_input_quantizer') or hasattr(mod_sub, '_weight_quantizer'):
                        set_quantizers(name_mod, mod_sub, **kwargs)
