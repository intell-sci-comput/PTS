import pandas as pd

data_folder = 'data/'

def main():
    xl = pd.ExcelFile(data_folder + 'Ball_drops_data.xls')
    all_balls = xl.sheet_names

    for task in all_balls:

        data = pd.read_excel(data_folder + 'Ball_drops_data.xls', task)

        train_sample = data[(data['Drop #'] == 1) &
                            (data['Time (s)'] <= 2)][['Time (s)', 'Height (m)']]
        train_sample.to_csv(data_folder + task + '_train.csv', index=False, header=False)

        test_sample = data[(data['Drop #'] == 1) &
                           (data['Time (s)'] > 2)][['Time (s)', 'Height (m)']]
        test_sample.to_csv(data_folder + task + '_test.csv', index=False, header=False)


if __name__ == '__main__':
    main()