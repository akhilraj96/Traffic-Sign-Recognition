from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, val_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_path, test_path, val_path = data_transformation.initiate_data_transformation(
        train_data, test_data, val_data)

    model_trainer = ModelTrainer()
    print(model_trainer.train(train_path, val_path))
