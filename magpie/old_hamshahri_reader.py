import os


class OldHamshahriReader(object):

    def __init__(self, root='corpora/hamshahri'):
        self.root = root
        self.invalids = []

    def docs(self, count):
        docs = []
        c = 0
        limit_exceeded = False
        for root, dirs, files in os.walk(self.root):
            for name in files:
                if c == count:
                    limit_exceeded = True
                    break
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
                    doc['text'] = f.read().strip().decode('utf-8')
                    docs.append(doc)

                    f.close()

                except Exception as e:
                    print e
                    print('error in reading:', name)
                c += 1
            if limit_exceeded:
                break

        return docs

    def sklearn_docs(self, count):
        docs = []
        labels = []
        c = 0
        limit_exceeded = False
        for root, dirs, files in os.walk(self.root):
            for name in files:
                if c == count:
                    limit_exceeded = True
                    break
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
                    labels.append(catl[4:].strip())
                    docs.append(f.read().strip().decode('utf-8'))

                    f.close()

                except Exception as e:
                    print e
                    print('error in reading:', name)
                c += 1
            if limit_exceeded:
                break

        return docs, labels

