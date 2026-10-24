"""Create one owned native profile through the existing standalone profile owner."""
import argparse
import json
import os
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--root',type=Path,required=True)
args=parser.parse_args()
assert os.name=='nt'
home=args.root.resolve();source=args.source.resolve()
assert json.loads((home.parent/'.owner.json').read_text())['owner']=='windows-cloud-client-r1'
os.environ['HERMES_HOME']=str(home)
for key in ('HOME','USERPROFILE'):os.environ[key]=str(home.parent/'profile')
sys.path.insert(0,str(source))
from hermes_cli.profiles import create_profile
from hermes_cli.config import atomic_config_replace,read_user_config_raw
from hermes_constants import set_hermes_home_override,reset_hermes_home_override,get_default_hermes_root
beta=create_profile('beta',clone_config=True,no_alias=True)
assert not (beta/'.cloud-device.bin').exists()
assert not (beta/'.cloud-device.lock').exists()
token=set_hermes_home_override(str(beta))
try:
    config=read_user_config_raw()
    config['providers']['semantier-cloud'].pop('credential_ref',None)
    config['model'].update(default='',provider='semantier-cloud')
    config.get('billing',{}).pop('account_id',None)
    atomic_config_replace(beta/'config.yaml',config)
finally:reset_hermes_home_override(token)
print(json.dumps({'profile':'beta','owner_root':str(get_default_hermes_root()),'profile_home':str(beta),
    'protected_store_not_cloned':True,'source_only':not bool(os.environ.get('HERMES_QUALIFICATION_PAYLOAD')),'package_acceptance':False}))
