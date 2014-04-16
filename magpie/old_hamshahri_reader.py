import os


class OldHamshahriReader(object):

    def __init__(self, root='corpora/hamshahri'):
        self.root = root
        self.invalids = []

    def docs(self):
        docs = []
        for root, dirs, files in os.walk(self.root):
            for name in files:
                if name in self.invalids:
                    continue

                try:
                    f = open(os.path.join(root, name))
                    f.readline()
                    f.readline()
                    catl = f.readline()
                    if not '.Cat' in catl:
                        print('Doc doesnt have category, ', name)
                        continue
                    doc = {}
                    doc['cat'] = catl[4:].strip()
                    doc['text'] = f.read()
                    docs.append(doc)

                except Exception as e:
                        print('error in reading:', name)

        return docs
                   # for element in elements.getElementsByTagName('DOC'):
                            #doc = {}
                            #doc['id'] = element.getElementsByTagName('DOCID')[0].childNodes[0].data
                            #doc['issue'] = element.getElementsByTagName('ISSUE')[0].childNodes[0].data

                            #for cat in element.getElementsByTagName('CAT'):
                                    #doc['categories_'+ cat.attributes['xml:lang'].value] = cat.childNodes[0].data.split('.')

                            #doc['title'] = element.getElementsByTagName('TITLE')[0].childNodes[1].data
                            #doc['text'] = ''
                            #for item in element.getElementsByTagName('TEXT')[0].childNodes:
                                    #if item.nodeType == 4:  # CDATA
                                            #doc['text'] += item.data
                   #         yield doc
