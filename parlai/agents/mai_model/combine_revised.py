def combine_file1_to_file2(src_path, target_path):
    f = open(src_path, 'r')
    save_f = open(target_path, 'a')
    for line in f:
        temp_line = line.strip()
        save_f.write('\n%s' % temp_line)
    f.close()
    save_f.close()


def main():
    combine_file1_to_file2('data/train_self_revised.txt', 'data/train_self_original.txt')
    combine_file1_to_file2('data/valid_self_revised.txt', 'data/train_self_original.txt')


if __name__ == '__main__':
    main()