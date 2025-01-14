import data_collector
import app
import model_trainer

def main():
    user_input = input('Press [1] to Collect Data, [2] to Train Model, [3] to Recognize Signs:')
    
    if user_input == '1':
        collector = data_collector.DataCollector(folder='Data/C', img_size=300, offset=20)
        collector.collect_data()
    elif user_input == '2':
        trainer = model_trainer.ModelTrainer()
        trainer.train_model()
        pass
    elif user_input == '3':
        detector = app.App(img_size=30, offset=20)
        detector.detect_signs()
    else:
        print('Invalid input, exiting program.')

if __name__ == '__main__':
    main()