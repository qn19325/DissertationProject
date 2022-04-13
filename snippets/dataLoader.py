class ImageDataLoader(Dataset):
    def __init__(self):
        self.data_df = pandas.read_csv(csvFilename)
        self.dataset_len = len(self.data_df)
    def __getitem__(self, idx):
        f_name_t = self.data_df['Filename'][idx]
        f_name_tp1 = self.data_df['Filename'][idx+1]
        label = self.data_df['Label'][idx]
        label = label.astype(np.float32) 
        label = np.true_divide(label, 10) # normalise speeds
        img_t = torchvision.io.read_image('trainingData/{}'.format(f_name_t))
        img_tp1 = torchvision.io.read_image('trainingData/{}'.format(f_name_tp1))
        img_t = img_t.flatten().float().div_(255.0) # normalize pixel values
        img_tp1 = img_tp1.flatten().float().div_(255.0) # normalize pixel values
        input_seq = torch.cat((img_t, img_tp1), dim=0)
        return input_seq, label
    def __len__(self):
        return self.dataset_len - 1

class SequentialImageDataLoader(Dataset):
    def __init__(self):
        self.data_df = pandas.read_csv(csvFilename)
        self.dataset_len = len(self.data_df) 
    def __getitem__(self, idx):
        f_name_t = self.data_df['Filename'][idx]
        f_name_tp1 = self.data_df['Filename'][idx+1]
        label = self.data_df['Label'][idx]
        label = label.astype(np.float32) 
        label = np.true_divide(label, 10) # normalise speeds
        img_t = torchvision.io.read_image('training/{}'.format(f_name_t))
        img_tp1 = torchvision.io.read_image('training/{}'.format(f_name_tp1))
        img_t = img_t.float().div_(255.0) # normalize pixel values
        img_tp1 = img_tp1.float().div_(255.0) # normalize pixel values
        return img_t, img_tp1, label
    def __len__(self):
        return self.dataset_len - 1

