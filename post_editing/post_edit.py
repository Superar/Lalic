import threading
from multiprocessing import cpu_count
from read_muse_embeds import load_embeddings, closest_words


class PostEditor(threading.Thread):

    def __init__(self, window, blast_reader, emb_en, emb_pt, progress_var, queue):
        threading.Thread.__init__(self)
        self.window = window
        self.blast_reader = blast_reader
        self.emb_en = emb_en
        self.emb_pt = emb_pt
        self.progress_var = progress_var
        self.queue = queue

        self.start()

    def run(self):
        errors = self.blast_reader.get_filtered_errors(
            [self.window.error_type.get()])
        
        save_file_content = ''
        save_file_content += '@annotations\n'
        save_file_content += str(self.window.app.cur_line)
        save_file_content += '\n'
        error_num = 0

        for error in errors:
            if self.window.stop:
                self.queue.put(-1)
                break

            self.progress_var.set(error_num)
            line = error[0]
            save_file_content += ' '.join(self.blast_reader.src_lines[line])
            save_file_content += '\n'
            save_file_content += ' '.join(self.blast_reader.ref_lines[line])
            save_file_content += '\n'
            save_file_content += ' '.join(self.blast_reader.sys_lines[line])
            save_file_content += '\n'

            error_info = [','.join(map(str, e)) for e in error[1][:-1]]
            error_info.append(error[1][-1])
            save_file_content +='#'.join(error_info)
            save_file_content += '\n'

            sentence_to_correct = self.blast_reader.src_lines[line]
            sys_sentence = self.blast_reader.sys_lines[line]
            candidates = list()
            for i in error[1][0]:
                if i > 0:
                    candidates.extend(['-.-'.join([w[0], 'white']) for w in closest_words(
                        sentence_to_correct[i], self.emb_en, self.emb_pt,
                        words_to_ignore=[sys_sentence[j] for j in error[1][1]])])
                else:
                    candidates.append('-.-'.join(['***', 'white']))
            save_file_content += '#@'.join(candidates)
            save_file_content += '\n'
            error_num = error_num + 1
        
        if not self.window.stop:
            save_file = open(self.window.filename, 'w')
            save_file.write(save_file_content)
            save_file.close()
            self.queue.put(0)
