from model import NCF
import tensorflow as tf
import torch
import sys
from torch.utils.data import Dataset, DataLoader
import pandas as pd



inp = input('Insert Data Path : ')
data = pd.read_csv(inp)
data = data.head(10)
data = data.drop(['u', 'i', 'date'], axis=1)
user_list = data.user_id.unique()
item_list = data.recipe_id.unique()
user2id = {w: i for i, w in enumerate(user_list)}
item2id = {w: i for i, w in enumerate(item_list)}

class Ratings_Datset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index()

    def __len__(self):
        return len(self.df)
  
    def __getitem__(self, idx):
        user = user2id[self.df['user_id'][idx]]
        user = torch.tensor(user, dtype=torch.long)
        item = item2id[self.df['recipe_id'][idx]]
        item = torch.tensor(item, dtype=torch.long)
        rating = torch.tensor(self.df['rating'][idx], dtype=torch.float)
        return user, item, rating



testloader = DataLoader(Ratings_Datset(data), batch_size=64, num_workers=2)


n_user = 226570
n_items = 231637
model = NCF(n_user, n_items).cuda()
model.load_state_dict(torch.load('models/mymodel.h5'))



users, recipes, r = next(iter(testloader))
users = users.cuda()
recipes = recipes.cuda()
r = r.cuda()

y = model(users, recipes)*5
print("ratings", r[:10].data)
print("predictions:", y.flatten()[:10].data)


