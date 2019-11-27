#encoding=utf-8
import argparse
import os
import codecs
import re

parser = argparse.ArgumentParser()
parser.add_argument("-train", type=str, default='F:/BaiduNetdiskDownload/train')
parser.add_argument("-dev", type=str, default='F:/BaiduNetdiskDownload/dev')
args = parser.parse_args()

entity_left_tag = '['
entity_right_tag = ']/'

WS_TAGS = {'b-wd':0, 'i-wd':1, 'e-wd':2, 's-wd':3}
wstags = set()
postags = set()
entitytags = set()

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

class DataGenerator():
    def __init__(self, level=1):
        self.level = level #0 word seg,  1  pos, 2 name entity
        self.buildtag = True
    def get_train(self):
        self.loaddata(path=args.train)

    def get_dev(self):
        self.loaddata(path=args.dev)

    def loaddata(self, path):
        for t in os.walk(path):
            d, s, files = t
            for i, f in enumerate(files):
                ff = os.path.join(d, f)
                #print('path', i, ff)
                self.loadtext(os.path.join(d, ff))
    def loadtext(self, fpath):
        for line in codecs.open(fpath, 'r', 'utf-8'):
            word_tags = line.strip().split()
            if self.level == 0:
                chars = []
                tags  = []
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
                        pos = ['s-' + 'wd']
                    elif l == 2:
                        pos = ['b-' + 'wd', 'e-' + 'wd']
                    elif l > 2:
                        pos = ['b-' + 'wd'] + [ 'i-' + 'wd' for _ in range(l-2)] + ['e-' + 'wd']
                    else:
                        pass
                    if self.buildtag is True:
                        wstags.update(set(pos))
                    chars.extend(wd)
                    tags.extend(pos)
                print(chars)
                print([WS_TAGS[t] for t in tags])
            elif self.level == 1:
                chars = []
                tags  = []
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
                        pos = ['s-' + pos]
                    elif l == 2:
                        pos = ['b-' + pos, 'e-' + pos]
                    elif l > 2:
                        pos = ['b-' + pos] + ['i-' + pos for _ in range(l - 2)] + ['e-' + pos]
                    if self.buildtag is True:
                        postags.update(set(pos))
                    chars.extend(wd)
                    tags.extend(pos)
                #print(chars)
                #print(tags)
            elif self.level == 2:
                enter_entity = False
                entity_tag = 'o'
                entity_slot = []
                words = []
                tags = []
                for i, wordtag in enumerate(word_tags):
                    if wordtag[0] == entity_left_tag:
                        if wordtag[1] == '/':
                            pass
                        else:
                            #entity begin
                            wordtag = wordtag[1:]
                            enter_entity = True
                    #entity end
                    if wordtag.find(entity_right_tag) > 0:
                        enter_entity = False
                        entity_tag = wordtag[wordtag.find(']/')+2:]
                        entity_slot.append(wordtag[0:wordtag.find('/')])

                        l = len(entity_slot)
                        if l == 1:
                            entity_tag = ['s-' + entity_tag]
                        elif l == 2:
                            entity_tag = ['b-' + entity_tag, 'e-' + entity_tag]
                        elif l > 2:
                            entity_tag = ['b-' + entity_tag] + ['i-' + entity_tag for _ in range(l - 2)] + ['e-' + entity_tag]

                        if self.buildtag is True:
                            entitytags.update(set(entity_tag))
                        #print(entity_slot, entity_tag)
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
                            pos = 'w'
                        else:
                            wd = wordtag[0:ws_pos]
                            pos = wordtag[ws_pos + 1:]
                        #print(wd, 'o')
                        words.append(wd)
                        tags.append('o')
                #print(words)
                #print(tags)
                if len(words) != len(tags):
                    print(len(words), len(tags))


if __name__ == '__main__':
    datagenerator = DataGenerator(level=0)
    datagenerator.get_train()
    print(len(wstags), sorted(list(wstags), key=lambda x:x[2:]))
    print(len(postags), sorted(list(postags), key=lambda x:x[2:]))
    print(len(entitytags), sorted(list(entitytags), key=lambda x:x[2:]))
    datagenerator.get_dev()
