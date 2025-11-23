from tdc.single_pred import Tox, ADME

def load_herg():
    data = Tox(name="hERG")
    df = data.get_data()
    df = df.rename(columns={"Drug": "smiles", "Y": "label"})
    return df

def load_caco2():
    data = ADME(name="Caco2_Wang")
    df = data.get_data()
    df = df.rename(columns={"Drug": "smiles", "Y": "label"})
    return df

def load_ld50():
    data = Tox(name="LD50")
    df = data.get_data()
    df = df.rename(columns={"Drug": "smiles", "Y": "label"})
    return df
