import os,json 
from os.path import join,isfile
import librosa
from pydub import AudioSegment
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write



class AudioLoad():
    '''A CLASS FOR LOADINDING AUDIO FILES AND GET MFCCs FEATURES'''
    def __init__(self,fpath,sample_rate=None,dur=None,label=None):
        self.path = fpath
        self.signal, self.sample_rate = librosa.load(self.path,duration=dur,sr=sample_rate)
        self.label = label

    
    def get_mfcc(self,n_mfcc=13,n_fft=2048,hop_length=512,sample_to_cnsd=22050):
        if len(self.signal) >= sample_to_cnsd:
            target_signal=self.signal[:sample_to_cnsd] 
            MFCCs=librosa.feature.mfcc(target_signal, self.sample_rate, n_mfcc=n_mfcc, n_fft=2048,hop_length=512)                            
            return MFCCs

    def isValid_to_that_mfcc(self,n_mfcc=13,n_fft=2048,hop_length=512,sample_to_cnsd=22050):
        mfccs = self.get_mfcc(n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length,sample_to_cnsd=sample_to_cnsd)
        state = True if mfccs is not None else False 
        return (True,mfccs) if state else (False,None)













class AudioTransform(AudioLoad):
    '''A CLASS FOR AUGMENTING AUDIO DATA'''
    def __init__(self,fpath):
        super().__init__(fpath)

    #! require librosa implemention
    def add_silent(self,silent_dur=500,after=True,mkwav=True,output_path=None):
        # create silent_dur msec of silence audio segment
        dur_sec_segment = AudioSegment.silent(duration=silent_dur)  #duration in milliseconds
        in_audio_file = AudioSegment.from_wav(self.signal) 
        final_audio_file = (in_audio_file + dur_sec_segment) if after else (dur_sec_segment + in_audio_file)
        output_path = 'output.wav' if output_path is None else output_path
        if mkwav:final_audio_file.export(output_path, format="wav")
        else: return final_audio_file 

    def change_Pitch(self,pitch_factor=None,mkwav=True,output_path=None):
        pitch_factor = np.random.uniform(0.7,1.7) if pitch_factor is None else pitch_factor
        augmented_data = librosa.effects.pitch_shift(self.signal, self.sample_rate, pitch_factor)
        print(f'augmented with pitch_factor : {pitch_factor}')
        output_path = 'pitched_output.wav' if output_path is None else output_path
#         if mkwav:augmented_data.export(output_path, format="wav")
        if mkwav:write(output_path,self.sample_rate,data=augmented_data)
        else: return augmented_data

    def change_Speed(self,speed_factor=None,mkwav=True,output_path=None):
        speed_factor = np.random.uniform(0.7,1.7) if speed_factor is None else speed_factor
        augmented_data = librosa.effects.time_stretch(self.signal, speed_factor)
        print(f'augmented with speed_factor : {speed_factor}')
        output_path = 'speeded_output.wav' if output_path is None else output_path
#         if mkwav:augmented_data.export(output_path, format="wav")
        if mkwav:write(output_path,self.sample_rate,data=augmented_data)
        else: return augmented_data











def load_dfs(directory_path,batch=None,as_gen=False):
    if os.path.exists(directory_path):
        if as_gen:
            files = (join(directory_path,fname) for fname in os.listdir(directory_path) if isfile(join(directory_path,fname)))
            return files
        else:
            files = [join(directory_path,fname) for fname in os.listdir(directory_path) if isfile(join(directory_path,fname))]
            return files[:batch]
        
    print("the directory path you provided doesn't exists")
    return None
    



def audioData_to_jsonLoaderV2(dataSetPath,jsonPath=None):
    #saving prossing data to dictionary
    dataDict = {
        "mapping":[],#["snap","notSnap"]
        "labels":[],#[0,1,1,0,1]
        "MFCC":[],#mfccVectors
        "files":[]}#files dicritory
    
    for i ,(dirPath,_,fileNames) in enumerate(os.walk(dataSetPath)):
        # we need to ensure we are not at root level
        if dirPath is not dataSetPath:
            category=os.path.split(dirPath)#data/snap ->"data" ,"snap"
            dataDict["mapping"].append(category)
            print(f"processing {category} ......")
            for file in fileNames:
                #get file Path
                file_Path=os.path.join(*category,file)
                #load audio files 
                state,mfccs = AudioLoad(file_Path,sample_rate=22050).isValid_to_that_mfcc()
                #ensure the audio file at least 1 sec
                if state:
                    # load the data to the dict
                    dataDict["labels"].append(i-1)
                    dataDict["MFCC"].append(mfccs.T.tolist())
                    dataDict["files"].append(file_Path)
    jsonPath = "audiodataset.json" if jsonPath is None else jsonPath
    #Store the dict to json file
    with open(jsonPath,'w+') as fp:
        print("")
        json.dump(dataDict,fp)   

def load_thatJsonData(dataPath):
    #open the data json file 
    with open(dataPath,'r') as fp:
        data=json.load(fp)

    X =np.array(data['MFCC'])
    y =np.array(data['labels'])
    label_map= data['mapping'][-1]
    return X, y, label_map



















#/=========================== OLD MODULE ==================================/

def data_to_jsonLoader(files,jsonPath="dataset.json",encode=1):
    dataDict = {
        "labels":[],#[0,1,1,0,1] 1=>snap
        "MFCC":[],#mfccVectors   
    }    
    for fn in files:
        mfcc = preprocces_toMfcc(fn)
        if valid(mfcc):
            dataDict["labels"].append(encode)
            dataDict["MFCC"].append(mfcc.T.tolist())
            print(f'{fn}....has been completed and processed')

    with open(jsonPath,'w+') as fp:
        print('loading Data to Json.....')
        json.dump(dataDict,fp)
        print("DONE")

def add_silent(input_fname,dur=500,after=True,output_fname='outfile.wav'):
    # create 1 sec of silence audio segment
    dur_sec_segment = AudioSegment.silent(duration=dur)  #duration in milliseconds
    #read wav file to an audio segment
    in_audio_file = AudioSegment.from_wav(input_fname)
    #Add above two audio segments    
    final_audio_file = (in_audio_file + dur_sec_segment) if after else (dur_sec_segment + in_audio_file)
    #? Either save modified audio
    final_audio_file.export(output_fname, format="wav")


def aug_thataudio(input_fname,noise_in_file = False,change_pitch = False,change_speed = False,pitch_factor = None,speed_factor = None,make_wav=True,**aug_par):
    '''adding noise ,change_pitch, change_speed,pitch_factor,speed_factor'''
    
    nameState = 'noised' if noise_in_file else ('pitched' if change_pitch else 'speeded')
    output_fname = f"{input_fname[:-4]}_{nameState}Augmented.wav"
    if pitch_factor is None or speed_factor is None:pitch_factor = np.random.uniform(0.7,2.7);speed_factor = np.random.uniform(0.7,1.7)
        
    # loading the the audio file 
    data,sample_rate = librosa.load(input_fname,sr=None)
    # introduce noise to the audio file
    if noise_in_file:
        noise_factor = np.random.uniform(0.0,0.01)
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        print(f'pitch_factor : {pitch_factor}')                              
        # Cast back to same data type
        augmented_data = augmented_data.astype(type(data[0]))
    if change_pitch:
        augmented_data = librosa.effects.pitch_shift(data, sample_rate, pitch_factor,)
        print(f'augmented with pitch_factor : {pitch_factor}')
    if change_speed:  
        augmented_data = librosa.effects.time_stretch(data, speed_factor)
        print(f'augmented with speed_factor : {speed_factor}')
    if make_wav:sf.write(output_fname,augmented_data,sample_rate,format='wav')
    
    return output_fname,augmented_data

   
def valid(mfcc):
    return True if mfcc is not None else False


def count_valid(files):
    v_files,n_files=0,0
    for fn in files:
        mfcc = preprocces_toMfcc(fn)
        v_files= v_files+1 if valid(mfcc) else v_files
        n_files += 1
    return f'valid_files: {v_files} ,all_files: {n_files}'


def preprocces_toMfcc(file_path,sample_to_cnsd=22050,n_mfcc=13):
        signal,sr=librosa.load(file_path)
        if len(signal) >= sample_to_cnsd:
            target_signal=signal[:sample_to_cnsd] 
            MFCCs=librosa.feature.mfcc(target_signal, sr, n_mfcc=n_mfcc, n_fft=2048,hop_length=512)                            
            return MFCCs
