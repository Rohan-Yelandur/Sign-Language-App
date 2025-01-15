import data_collector
import app
import model_trainer

def main():
    img_size = 224 # 224 because MobileNetV2 requires input images to be 224x224

    user_input = input('Press [1] to Collect Data, [2] to Train Model, [3] to Recognize Signs:')
    if user_input == '1':
        collector = data_collector.DataCollector(data_folder='Data/C', img_size=img_size, offset=20)
        collector.collect_data()
    elif user_input == '2':
        trainer = model_trainer.ModelTrainer(data_dir='Data', model_save_path='Model/keras_model_v3.h5', img_size=img_size, batch_size=32, epochs=20)
        trainer.train_model()
        pass
    elif user_input == '3':
        detector = app.App(img_size=30, offset=20, model_path='Model/keras_model_v3.h5', labels_path='Model/labels.txt')
        detector.detect_signs()
    else:
        print('Invalid input, exiting program.')

if __name__ == '__main__':
    main()