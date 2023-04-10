import numpy as np
import opensmile
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

def get_dataset():
    samples = ['300','301','302','303','304','305','306','307','308','309','310','311','312','313','314','315','316','317','318','319','320','321','322','323','324','325','326','327','328','329','330','331','332','333','334','335','336','337','338','339','340','341','343','344','345','346','347','348','349','350','351','352','353','354','355','356','357','358','359','360','361','362','363','364','365','366','367','368','369','370','371','372','373','374','375','376','377','378','379','380','381','382','383','384','385','386','387','388','389','390','391','392','393','395','396','397','399','400','401','402','403','404','405','406','407','408','409','410','411','412','413','414','415','416','417','418','419','420','421','422','423','424','425','426','427','428','429','430','431','432','433','434','435','436','437','438','439','440','441','442','443','444','445','446','447','448','449','450','451','452','453','454','455','456','457','458','459','461','462','463','464','465','466','467','468','469','470','471','472','473','474','475','476','477','478','479','480','481','482','483','484','485','486','487','488','489','490','491','492']
    datadir = '../../../../Data/DAICWoz/'
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

    dataset = {}
    for sample in tqdm(samples):
        audio = []
        visual = []

        filename = datadir + sample + '_AUDIO.wav'
        y = smile.process_file(filename)
        indexes = [int(i[1].total_seconds() * 1e2) for i in y.index]
        y = y.to_numpy()


        filename = datadir + sample + '_CLNF_features.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]

        for line in lines:
            fields = line.strip().split(',')
            time = float(fields[1])        
            confidence = float(fields[2])

            if confidence < 0.01:
                continue

            time = int(time * 1e2)
            if time not in indexes:
                continue

            feature = [float(i) for i in fields[4:]]
            visual.append(feature)

            index =  indexes.index(time)
            audio.append(y[index,:])

        visual = np.array(visual)
        audio = np.array(audio)
        dataset[sample] = (visual, audio)

    with open("./daic.pickle", 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def trim(a, v):
    leng = min(a.shape[0], v.shape[0])
    return a[:leng, :], v[:leng, :]


class DAICWOZ(Dataset):
    def __init__(self, dataset, filename, is_train=True, maxlen=1024):
        super(DAICWOZ, self).__init__()
        self.dataset = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]

        for line in (lines):
            sample, label = line.strip().split(',')[:2]
            label = int(label)
            
            if is_train == False:
                a = dataset[sample][1][0::30, :]
                v = dataset[sample][0][0::30, :]
                a = a[-maxlen:,:]
                v = v[-maxlen:,:]
                self.dataset.append((a, v, label))
                continue

            if label == 0:
                for j in range(11):
                    a = dataset[sample][1][j::30, :]
                    v = dataset[sample][0][j::30, :]
                    a = a[-maxlen:,:]
                    v = v[-maxlen:,:]
                    self.dataset.append((a, v, label))
            else:
                for j in range(30):
                    a = dataset[sample][1][j::30, :]
                    v = dataset[sample][0][j::30, :]
                    a = a[-maxlen:,:]
                    v = v[-maxlen:,:]
                    self.dataset.append((a, v, label))
                
        
        self.length = [d[0].shape[0] for d in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.dataset[idx][2], self.length[idx]

class MultiviewDAICWOZ(Dataset):
    def __init__(self, dataset, filename, is_train=True, maxlen=1024):
        super(MultiviewDAICWOZ, self).__init__()
        self.dataset = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]

        for line in (lines):
            sample, label = line.strip().split(',')[:2]
            label = int(label)
            
            if is_train == False:
                a = dataset[sample][1][0::30, :]
                v = dataset[sample][0][0::30, :]
                a = a[-maxlen:,:]
                v = v[-maxlen:,:]
                self.dataset.append((a, v, label))
                continue

            audio, video = [], []
            if label == 0:
                for j in range(11):
                    a = dataset[sample][1][j::30, :]
                    v = dataset[sample][0][j::30, :]
                    a = a[-maxlen:,:]
                    v = v[-maxlen:,:]
                    audio.append(a)
                    video.append(v)
                    if (j % 2) == 1:
                        audio = trim(audio[0], audio[1])
                        video = trim(video[0], video[1])
                        self.dataset.append((audio, video, label))
                        audio, video = [], []
            else:
                for j in range(30):
                    a = dataset[sample][1][j::30, :]
                    v = dataset[sample][0][j::30, :]
                    a = a[-maxlen:,:]
                    v = v[-maxlen:,:]
                    audio.append(a)
                    video.append(v)
                    if (j % 2) == 1:
                        audio = trim(audio[0], audio[1])
                        video = trim(video[0], video[1])
                        self.dataset.append((audio, video, label))
                        audio, video = [], []
                
        
        self.length = [d[0][0].shape[0] for d in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.dataset[idx][2], self.length[idx]
