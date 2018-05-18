class ApeReader(object):

    def __init__(self, filename):
        self.error_lines = list()
        self.src_lines = list()
        self.ref_lines = list()
        self.sys_lines = list()
        self.corrections = list()
        self.filename = filename
        self.cur_line = -1

        with open(filename, 'r') as _file:
            if not _file.readline().strip() == '@annotations':
                raise RuntimeError('Formato inválido')

            self.cur_line = int(_file.readline().strip())

            while True:
                src = _file.readline().split()
                if not src:
                    break

                self.src_lines.append(src)
                self.ref_lines.append(_file.readline().split())
                self.sys_lines.append(_file.readline().split())
                error = _file.readline().split('#')

                error_indexes = [list(map(int, e.split(',')))
                                 for e in error[:-1]]
                error_indexes.append(error[-1])
                self.error_lines.append(error_indexes)

                cor = _file.readline().strip().split('#@')
                self.corrections.append([c.split('-.-') for c in cor])

    def save(self):
        with open(self.filename, 'w') as _file:
            _file.write('@annotations\n')
            _file.write(str(self.cur_line))
            _file.write('\n')

            for line in range(len(self.src_lines)):
                _file.write(' '.join(self.src_lines[line]))
                _file.write('\n')
                _file.write(' '.join(self.ref_lines[line]))
                _file.write('\n')
                _file.write(' '.join(self.sys_lines[line]))
                _file.write('\n')

                error_info = [','.join(map(str, e))
                              for e in self.error_lines[line][:-1]]
                error_info.append(self.error_lines[line][-1])
                _file.write('#'.join(error_info))

                _file.write('#@'.join(['-.-'.join(w)
                                       for w in self.corrections[line]]))
                _file.write('\n')
