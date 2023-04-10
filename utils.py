import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# First of all, we need to define the function that counting number of description and name of description.

def plot_freq_top10(careplan_df, patient_list, csv_name, code_name='CODE', desc_name='DESCRIPTION'):
    # Extract the rows with patient list
    filtered_careplan = careplan_df[careplan_df['PATIENT'].isin(patient_list)]

    # Group the data by code and description and count the frequency in the careplans
    grouped_careplan = filtered_careplan.groupby([code_name, desc_name]).size().reset_index(name='Freq')

    # Sort the data by frequency in descending order and select top 10
    top_10_data = grouped_careplan.sort_values('Freq', ascending=False).head(10)

    # Plot the top 10 data
    top_10_data.plot.barh(x=desc_name, y='Freq', figsize=(8, 5))
    plt.ylabel(csv_name)
    plt.show()

# First of all, we need to define the function that counting number of description and name of description.

def plot_freq_top5(careplan_df, patient_list, csv_name, code_name='CODE', desc_name='DESCRIPTION'):
    # Extract the rows with patient list
    filtered_careplan = careplan_df[careplan_df['PATIENT'].isin(patient_list)]

    # Group the data by code and description and count the frequency in the careplans
    grouped_careplan = filtered_careplan.groupby([code_name, desc_name]).size().reset_index(name='Freq')

    # Sort the data by frequency in descending order and select top 5
    top_5_data = grouped_careplan.sort_values('Freq', ascending=False).head(5)

    # Plot the top 10 data
    top_5_data.plot.barh(x=desc_name, y='Freq', figsize=(8, 5))
    plt.ylabel(csv_name)
    plt.show()

def calculateAge(birthDate, endDate):
    return endDate.year - birthDate.year - ((endDate.month, endDate.day) < (birthDate.month, birthDate.day)) 

def get_df(csv_base_path, csv_name, patients_list, column='PATIENT'):
    df = pd.read_csv(csv_base_path + csv_name + ".csv")
    df = df[df[column].isin(patients_list)]
    return df

def get_patient_and_code_df(df, column):
    temp_df = pd.DataFrame(column)
    for i in range(len(df)) :
        # Bring patient's ID and Code
        temp_df.loc[i] = [df.iloc[i]["PATIENT"], df.iloc[i]["CODE"]]
    
    return temp_df

def generate_patient_dataframe(csv_base_path, patients_list):
  
    # From the patients.csv file, we need extract the unique patient's ID
    patients_df = get_df(csv_base_path, 'patients', patients_list, 'Id')

    # Update the unique patient's ID list
    patients_list = patients_df['Id'].to_list()
    print(len(patients_list))

    # Change the Birthdate and Deathdate column dtype from string to datetime to calculate age
    patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'], format='%Y/%m/%d')    # 1
    patients_df['DEATHDATE'] = pd.to_datetime(patients_df['DEATHDATE'], format='%Y/%m/%d')    # 1
    # If DEATHDATE is null, replace with current date
    patients_df['DEATHDATE'] = patients_df['DEATHDATE'].fillna(pd.to_datetime(pd.Timestamp.now(), format='%Y/%m/%d'))
    # patients_df
    columns = {"pat_id": [], "Age": [], "Gender" :[], "Ethnic" :[], "Race" : []}
    pat_df = pd.DataFrame(columns)
    for i in range(len(patients_df)) :
        # Bring patient's ID, age and Gender
        pat_df.loc[i] = [patients_df.iloc[i]["Id"]  , calculateAge(patients_df.iloc[i]['BIRTHDATE'], patients_df.iloc[i]['DEATHDATE']), 
                        patients_df.iloc[i]["GENDER"], patients_df.iloc[i]["ETHNICITY"], patients_df.iloc[i]["RACE"]]
    pat_df.loc[pat_df['Gender'] == 'M', 'Gender'] = 0
    pat_df.loc[pat_df['Gender'] == 'F', 'Gender'] = 1
    pat_df.loc[pat_df['Ethnic'] == 'hispanic', 'Ethnic'] = 0
    pat_df.loc[pat_df['Ethnic'] == 'nonhispanic', 'Ethnic'] = 1
    pat_df.loc[pat_df['Race'] == 'asian', 'Race'] = 0
    pat_df.loc[pat_df['Race'] == 'black', 'Race'] = 1
    pat_df.loc[pat_df['Race'] == 'native', 'Race'] = 2
    pat_df.loc[pat_df['Race'] == 'other', 'Race'] = 3
    pat_df.loc[pat_df['Race'] == 'white', 'Race'] = 4
    
    return pat_df

def get_patients_list(csv_base_path, csv_name, column, top_num):
   
    # Prepare the dataframe    
    df = pd.read_csv(csv_base_path + csv_name + ".csv")    
    grouped_data = df.groupby(column).size().reset_index(name='Freq')

    # Check top highest careplans and code
    top_data = grouped_data.sort_values('Freq', ascending=False).head(top_num)
    top_list = top_data[column[0]].to_list()    
    print("Top " + str(csv_name) + " code : ",top_list)    

    # Based on the careplan code, extract the patients id
    df_sort = df[df[column[0]].isin(top_list)]
    patients_list = np.unique(df_sort['PATIENT'])

    # In the careplan, there are duplicated patient's ID 
    print("Number of patients: ", len(patients_list))

    return patients_list, top_list


def preprocess_medication_df(medications_df):
    # Change the START and STOP column dtype from string to datetime to calculate age
    medications_df['START'] = pd.to_datetime(medications_df['START'], format='%Y/%m/%d')    # 1
    medications_df['STOP'] = pd.to_datetime(medications_df['STOP'], format='%Y/%m/%d')    # 1
    # If STOP is null, replace with current date
    medications_df['STOP'] = medications_df['STOP'].fillna(pd.to_datetime(pd.Timestamp.now(), format='%Y/%m/%d'))
    return medications_df

def get_meditation_dataframe(csv_base_path, patients_list):
    
    medications_df = get_df(csv_base_path, 'medications', patients_list)
    medications_df = preprocess_medication_df(medications_df)    

    # New dataframe for medication
    column = {"pat_id": [], "med_code": []}    
    med_df = get_patient_and_code_df(medications_df, column)    
    # Remove duplicates
    med_df = med_df.drop_duplicates(keep='first', ignore_index=True)
    
    return med_df

def get_meditation_dataframe_v2(csv_base_path, patients_list):
    
    medications_df = get_df(csv_base_path, 'medications', patients_list)
    medications_df = preprocess_medication_df(medications_df)    
    
    # New dataframe for medication
    med_df = pd.DataFrame({"pat_id": [], "med_code": [], "med_day" :[]})
    for i in range(len(medications_df)) :
        # Bring patient's ID , medication code and how long patient use this medication
        med_df.loc[i] = [medications_df.iloc[i]["PATIENT"], medications_df.iloc[i]["CODE"], medications_df.iloc[i]['STOP'].replace(tzinfo=None)  - medications_df.iloc[i]['START'].replace(tzinfo=None)]
    
    med_df['med_day'] = med_df['med_day'].dt.days.astype('int16')
    
    # Some case has same patient, same medication in different period.
    # But in this case, we sum up the days where patient and medication are same even thought it's different time period.
    med_df = med_df.drop_duplicates(keep='first', ignore_index=True)
    
    return med_df

def gen_new_dataframe(csv_base_path, csv_name, column, patients_list):
    
    dataFrame = get_df(csv_base_path, csv_name, patients_list)
    df = pd.DataFrame({
        list(column.keys())[0]: dataFrame['PATIENT'],
        list(column.keys())[1]: dataFrame['CODE']
    })    
    return df

def generate_df_ymd(csv_base_path, csv_name, columns, patients_list):
    dataFrame = get_df(csv_base_path, csv_name, patients_list)
    dataFrame['DATE'] = pd.to_datetime(dataFrame['DATE'], format='%Y/%m/%d').dt.tz_localize(None)    # remove timezone
    df = pd.DataFrame({
        list(columns.keys())[0]: dataFrame['PATIENT'],
        list(columns.keys())[1]: dataFrame['CODE'],
        list(columns.keys())[2]: [date.year for date in dataFrame['DATE']],
        list(columns.keys())[3]: [date.month for date in dataFrame['DATE']],
        list(columns.keys())[4]: [date.day for date in dataFrame['DATE']]
    })
    return df