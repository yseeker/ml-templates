from src.recommend.ml.association_rule.fetcher import fetch_data_for_train
from src.recommend.ml.association_rule.trainer import Trainer
from src.recommend.ml.association_rule.production_model.config import parameter

if __name__ == "__main__":
    raw_df = fetch_data_for_train()
    trainer = Trainer(**parameter)
    trainer.train_and_save_model_for_production(raw_df=raw_df)
