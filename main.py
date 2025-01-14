import data_collector
import app

def main():
    user_input = input('Press [1] to COLLECT DATA, [2] to Recognize Signs: ')
    img_size = 300
    offset = 20
    folder = 'Data/A'
    
    if user_input == '1':
        collector = data_collector.DataCollector(folder, img_size, offset)
        collector.collect_data()
    elif user_input == '2':
        asl_detector = app.App(img_size, offset)
        asl_detector.detect_signs()
    else:
        print('Invalid input, exiting program.')

if __name__ == '__main__':
    main()