import jsonlines


def main():
    reader = jsonlines.open('checkpoint/fewnerd-mention_bio-bert_crf/result.jsonl')
    reader2 = jsonlines.open('checkpoint/fewnerd-mention_bio-bert_crf-finetune_base16/result.jsonl')
    seq_cnt, cur_id = 5, 0
    for item_hi, item_lo in zip(reader, reader2):
        assert item_hi['label'] == item_lo['label']
        label = item_hi['label']
        if item_hi['predict'] == label and item_lo['predict'] != label:
            print(item_hi)
            print(item_lo)
            assert len(item_hi['predict']) == len(item_lo['predict']) == len(item_hi['words'])
            for hi, lo, word in zip(item_hi['predict'], item_lo['predict'], item_hi['words']):
                if hi != lo:
                    print(word, end=' ')
            print()
            cur_id += 1
            if cur_id == seq_cnt:
                return


if __name__ == '__main__':
    main()
