#encoding=utf-8
from config import args
import os
import codecs


entity_left_tag = '['
entity_right_tag = ']/'

WS_TAGS = {'B-WD':0, 'I-WD':1, 'E-WD':2, 'S-WD':3, '[CLS]':4, '[SEP]':5, '[PAD]':6}
wstags = set()
postags = set()
entitytags = set()


class DataGenerator(object):
    def __init__(self, level=0, mode=0):
        self.level = level #0 word seg,  1  pos, 2 name entity
        self.buildtag = True
        self.mode=mode
        self.tags = set()
        self.wd_id = {}
        self.id_wd = {}
        self.load_vocab()

    def load_vocab(self):
        for idx, line in enumerate(codecs.open(args.vocab, 'r', 'utf-8')):
            self.wd_id[line.strip()] = idx
            self.id_wd[idx] = line.strip()

    def get_train(self, maxs=20000):
        if self.mode==0:
            self.loaddata(path=args.train, tarpath='./res_data/train.txt')
        elif self.mode == 1:
            return self.loaddata2(path=args.train, maxs=maxs)
    def get_dev(self, maxs=5000):
        if self.mode == 0:
            self.loaddata(path=args.dev, tarpath='./res_data/dev.txt')
        elif self.mode == 1:
            return self.loaddata2(path=args.dev, maxs=maxs)
    def loaddata2(self, path, maxs):
        words = []
        word_idxs = []
        tags  = []
        tag_idxs = []

        sentents_chs = []
        sentents = []
        sentents_tags = []
        segments_ids  = []
        masks = []
        for line in codecs.open(path, 'r', 'utf-8'):
            line = line.strip()
            if len(sentents) > maxs:
                break
            if len(line) == 0:
                words = ['[CLS]'] + words
                tags = ['[CLS]'] + tags
                masks.append(self.pad([1 for _ in words], args.maxlen, 0 ))

                words = self.pad(words, args.maxlen, '[PAD]')
                word_idxs = [self.wd_id.get(i, 100) for i in words] #100==[UNK]
                tags = self.pad(tags, args.maxlen, '[PAD]')
                tag_idxs = [WS_TAGS[i] for i in tags]

                sentents_chs.append(words)
                sentents.append(word_idxs)
                sentents_tags.append(tag_idxs)
                segments_ids.append([0 for _ in range(args.maxlen)])
                words = []
                word_idxs = []
                tags = []
                tag_idxs = []
            else:
                word, tag = line.strip().split()
                words.append(word)
                tags.append(tag)
        if len(words) > 0:
            words = ['[CLS]'] + words
            tags = ['[CLS]'] + tags
            masks.append(self.pad([1 for _ in words], args.maxlen, 0))
            words = self.pad(words, args.maxlen, '[PAD]')
            word_idxs = [self.wd_id.get(i, 100) for i in words]  # 100==[UNK]
            tags = self.pad(tags, args.maxlen, '[PAD]')
            tag_idxs = [WS_TAGS[i] for i in tags]
            sentents_chs.append(words)
            sentents.append(word_idxs)
            sentents_tags.append(tag_idxs)
            segments_ids.append([0 for _ in range(args.maxlen)])

        return sentents, masks, segments_ids, sentents_tags, sentents_chs

    @classmethod
    def pad(cls, vals, pad_size, pad_val):
        vals = vals[0:pad_size]
        if len(vals) < pad_size:
            vals = vals + [pad_val for i in range(pad_size - len(vals))]
        return vals

    def loaddata(self, path, tarpath):
        docs = []
        for t in os.walk(path):
            d, s, files = t
            for i, f in enumerate(files):
                ff = os.path.join(d, f)
                #print('path', i, ff)
                doc = self.loadtext(ff)
                docs.extend(doc)
        self.rewrite(docs, tarpath)

    def rewrite(self, doc, path):
        fw = codecs.open(path, 'w', 'utf-8')
        for i, line in enumerate(doc):
            chars = line[0]
            tags  = line[1]
            for j, ch in enumerate(chars):
                fw.write(ch + ' ' + tags[j] + '\n')
                if ch in ["。", '！', '？']:
                    fw.write('\n')
                else:
                    pass

    def loadtext(self, fpath):
        lines = []
        for line in codecs.open(fpath, 'r', 'utf-8'):
            word_tags = line.strip().split()
            if self.level == 0:
                chars, tags = self.read_wstag(word_tags, self.tags)
                lines.append([chars, tags])
            elif self.level == 1:
                chars, tags = self.read_postag(word_tags, self.tags)
                line.append([chars, tags])
            elif self.level == 2:
                chars, tags = self.read_entitytag(word_tags, self.tags)
                line.append([chars, tags])
        return lines

    def read_wstag(self, word_tags, wstags):
        chars = []
        tags = []
        for i, wordtag in enumerate(word_tags):
            if wordtag[0] == entity_left_tag:
                if wordtag[1] == '/':
                    pass
                else:
                    wordtag = wordtag[1:]
            if wordtag.find(entity_right_tag) > 0:
                wordtag = wordtag[0:wordtag.find(entity_right_tag)]
            ws_pos = wordtag.rfind('/')
            if ws_pos == -1:
                wd = wordtag
                pos = 'w'
            else:
                wd = wordtag[0:ws_pos]
                pos = wordtag[ws_pos + 1:]
            l = len(wd)
            if l == 1:
                pos = ['S-' + 'WD']
            elif l == 2:
                pos = ['B-' + 'WD', 'E-' + 'WD']
            elif l > 2:
                pos = ['B-' + 'WD'] + ['I-' + 'WD' for _ in range(l - 2)] + ['E-' + 'WD']
            else:
                pass
            if self.buildtag is True:
                wstags.update(set(pos))
            chars.extend(wd)
            tags.extend(pos)
        return chars, tags
    def read_postag(self, word_tags, postags):
        chars = []
        tags = []
        for i, wordtag in enumerate(word_tags):
            if wordtag[0] == entity_left_tag:
                if wordtag[1] == '/':
                    pass
                else:
                    wordtag = wordtag[1:]
            if wordtag.find(entity_right_tag) > 0:
                wordtag = wordtag[0:wordtag.find(entity_right_tag)]
            ws_pos = wordtag.rfind('/')
            if ws_pos == -1:
                wd = wordtag
                pos = 'W'
            else:
                wd = wordtag[0:ws_pos]
                pos = wordtag[ws_pos + 1:]
            l = len(wd)
            if l == 1:
                pos = ['S-' + pos]
            elif l == 2:
                pos = ['B-' + pos, 'E-' + pos]
            elif l > 2:
                pos = ['B-' + pos] + ['I-' + pos for _ in range(l - 2)] + ['E-' + pos]
            if self.buildtag is True:
                postags.update(set(pos))
            chars.extend(wd)
            tags.extend(pos)
        return chars, tags
    def read_entitytag(self, word_tags, entitytags):
        enter_entity = False
        entity_tag = 'O'
        entity_slot = []
        words = []
        tags = []
        for i, wordtag in enumerate(word_tags):
            if wordtag[0] == entity_left_tag:
                if wordtag[1] == '/':
                    pass
                else:
                    # entity begin
                    wordtag = wordtag[1:]
                    enter_entity = True
            # entity end
            if wordtag.find(entity_right_tag) > 0:
                enter_entity = False
                entity_tag = wordtag[wordtag.find(']/') + 2:]
                entity_slot.append(wordtag[0:wordtag.find('/')])

                l = len(entity_slot)
                if l == 1:
                    entity_tag = ['S-' + entity_tag]
                elif l == 2:
                    entity_tag = ['B-' + entity_tag, 'E-' + entity_tag]
                elif l > 2:
                    entity_tag = ['B-' + entity_tag] + ['I-' + entity_tag for _ in range(l - 2)] + ['E-' + entity_tag]

                if self.buildtag is True:
                    entitytags.update(set(entity_tag))
                # print(entity_slot, entity_tag)
                words.extend(entity_slot)
                tags.extend(entity_tag)
                entity_slot = []
                continue

            if enter_entity is True:
                entity_slot.append(wordtag[0:wordtag.rfind('/')])
            else:
                ws_pos = wordtag.rfind('/')
                if ws_pos == -1:
                    wd = wordtag
                    pos = 'W'
                else:
                    wd = wordtag[0:ws_pos]
                    pos = wordtag[ws_pos + 1:]
                # print(wd, 'o')
                words.append(wd)
                tags.append('O')
        return [words, tags]

if __name__ == '__main__':
    datagenerator = DataGenerator(mode=1)
    tups = datagenerator.get_train()
    print(tups[1])

    datagenerator.get_dev()
