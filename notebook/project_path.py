'''
Created on 30 May 2019

@author: nilesh
'''

import sys
import os

module_path = os.path.join(os.path.dirname(os.path.dirname(str(__file__))), "src")
resource_path = os.path.join(os.path.dirname(os.path.dirname(str(__file__))), "data")
config_path = os.path.join(os.path.dirname(os.path.dirname(str(__file__))), "conf")

_path = [module_path, resource_path, config_path]

for p in _path:
    if p not in sys.path:
        sys.path.append(p)
