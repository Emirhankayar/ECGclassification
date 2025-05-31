from pathlib import Path
import pandas as pd

# Set the path to the directory containing .hea files
directory = Path("C:/Users/faola/Downloads/physionet_data/training")


dx_list = ['426177001', '426783006', '164889003', '164890007', '427084000', '427393009', '426761007', '713422000']
dx_dict = {
    '426177001' : 'SB',
    '426783006' : 'SR',
    '164889003' : 'AFIB',
    '164890007' : 'AFIB',
    '427084000' : 'GSVT',
    '427393009' : 'SR',
    '426761007' : 'GSVT',
    '713422000' : 'GSVT'
}

label_dict = {
    "AFIB": 0,
    "GSVT": 1,
    "SB": 2,
    "SR": 3
}
'''
MAP to do

SB->SB
NSR->SR
AF->AFIB
AFL->AFIB
STach->GST
SA->SR
SVT->GSVT
ATach->GSVT
'''

df = pd.DataFrame(columns=['PatientID', 'Age', 'Gender', 'Dx', 'Rhythm', 'Label'])


def remove_useless_files_and_compile_df(dir, ls):
    hea_files = list(dir.rglob("*.hea"))
    mat_files = list(dir.rglob('*.mat'))

    for hea_file, mat_file in zip(hea_files, mat_files):
        if hea_file.stem != mat_file.stem:
            continue

        dx = None
        other_data = {}

        with hea_file.open('r') as file:
            for line in file:
                line = line.strip()

                if line.startswith("# Dx:"):
                    dx = line[len("# Dx:"):].strip()

                if line.startswith("# Age:"):
                    other_data['age'] = line[len("# Age:"):].strip()

                if line.startswith("# Sex:"):
                    other_data['sex'] = line[len("# Sex:"):].strip()

        if dx in ls:
            print(f"{hea_file.name}: Dx = {dx}, Other Info: {other_data}")
            df.loc[len(df)] = [hea_file.stem, other_data['age'], other_data['sex'], dx, dx_dict[dx], label_dict[dx_dict[dx]]]
        else:
            hea_file.unlink()
            mat_file.unlink()


        





remove_useless_files_and_compile_df(directory, dx_list)

df.to_excel("external_test_data.xlsx") 