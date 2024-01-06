import json

import torch
from torch.utils.data import DataLoader

from Modules.W2NER.Model import Model
from Utils import utils, MyDataset
from Utils.train import Trainer

config = json.load(
    open("./config.json", 'r', encoding='utf-8')
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config['device'] = DEVICE

logger = utils.get_logger(config['dataset'])
logger.info(config)
config['logger'] = logger


# random.seed(config.seed)
# np.random.seed(config.seed)
# torch.manual_seed(config.seed)
# torch.cuda.manual_seed(config.seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

logger.info("Loading Data")
datasets, ori_data = MyDataset.load_data_bert(config)

train_loader, test_loader = (
    DataLoader(dataset=dataset,
               batch_size=config['batch_size'],
               collate_fn=MyDataset.collate_fn,
               shuffle=i == 0,
               num_workers=0,
               drop_last=i == 0)
    for i, dataset in enumerate(datasets)
)

updates_total = len(datasets[0]) // config['batch_size'] * config['epochs']

logger.info("Building Model")
model = Model(config)

model = model.cuda()

trainer = Trainer(model, config, updates_total, logger)

best_f1 = 0
best_test_f1 = 0
for i in range(config['epochs']):
    logger.info("Epoch: {}".format(i))
    trainer.train(i, train_loader)
    f1 = trainer.eval(i, test_loader)
    test_f1 = trainer.eval(i, test_loader, is_test=True)
    if f1 > best_f1:
        best_f1 = f1
        best_test_f1 = test_f1
        # trainer.save(config.save_path)
logger.info("Best DEV F1: {:3.4f}".format(best_f1))
logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))

# trainer.load(config['save_path'])
# trainer.predict("Final", test_loader, ori_data[-1])
