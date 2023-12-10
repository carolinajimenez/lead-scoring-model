import os
from os import getenv
import shimoku_api_python as Shimoku

access_token = getenv('SHIMOKU_TOKEN')
universe_id: str = getenv('SHIMOKU_UNIVERSE_ID')
workspace_id: str = getenv('SHIMOKU_WORKSPACE_ID')

s = Shimoku.Client(
    access_token=access_token,
    universe_id=universe_id,
)
s.set_workspace(uuid=workspace_id)

s.set_board('Custom Board')

s.set_menu_path('catalog', 'bar-example')

language_expressiveness = [
    {'Language': 'C', 'Statements ratio': 1.0, 'Lines ratio': 1.0},
    {'Language': 'C++', 'Statements ratio': 2.5, 'Lines ratio': 1.0},
    {'Language': 'Fortran', 'Statements ratio': 2.0, 'Lines ratio': 0.8},
    {'Language': 'Java', 'Statements ratio': 2.5, 'Lines ratio': 1.5},
    {'Language': 'Perl', 'Statements ratio': 6.0, 'Lines ratio': 6.0},
    {'Language': 'Smalltalk', 'Statements ratio': 6.0, 'Lines ratio': 6.25},
    {'Language': 'Python', 'Statements ratio': 6.0, 'Lines ratio': 6.5},
]

s.plt.bar(
    order=0, title='Language expressiveness',
    data=language_expressiveness, x='Language',
    y=['Statements ratio', 'Lines ratio'],
)

# Necessary for notifying the front-end even if not using async execution
s.run()