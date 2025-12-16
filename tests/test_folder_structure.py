import os
from datetime import datetime

def test_folder_structure():
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join("logs", "imagens", date_str)
    assert os.path.exists(path), f"A pasta {path} não foi criada corretamente"

test_folder_structure()
