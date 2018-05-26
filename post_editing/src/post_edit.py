import threading
import queue
from multiprocessing import cpu_count
from readers.read_muse_embeds import closest_words


class PostEditor(threading.Thread):

    def __init__(self, window, blast_reader, progress_var):
        threading.Thread.__init__(self)
        self.window = window
        self.blast_reader = blast_reader
        self.emb_en = window.emb_en
        self.emb_pt = window.emb_pt
        self.progress_var = progress_var

        self.chunk_threads = list()

        self.msg_queue = window.ape_queue
        self.queue_threads_in = queue.Queue()
        self.queue_threads_out = queue.Queue()

        self.start()

    def run(self):
        errors = self.blast_reader.get_filtered_errors(
            [self.window.error_type.get()])

        save_file_content = ''
        save_file_content += '@annotations\n'
        save_file_content += str(self.window.app.cur_line)
        save_file_content += '\n'
        error_num = 0

        num_threads = cpu_count() - 1
        chunk_size = len(errors) // num_threads
        chunks = [list(enumerate(errors))[t * chunk_size:] if t == num_threads - 1 else
                  list(enumerate(errors))[t * chunk_size:(t+1) * chunk_size]
                  for t in range(num_threads)]

        for chunk in chunks:
            self.chunk_threads.append(PostEditChunk(self.blast_reader, self.emb_en, self.emb_pt,
                                                    chunk, self.queue_threads_in, self.queue_threads_out))

        content_list = ['' for _ in range(len(errors))]
        finished_threads = 0
        while True:
            if self.window.stop:
                for _ in range(num_threads):
                    self.queue_threads_in.put(-1)
                for thread in self.chunk_threads:
                    thread.join()
                break
            msg = self.queue_threads_out.get()
            if msg == 0:
                finished_threads += 1
            else:
                content_list[msg[0]] = msg[1]
                self.progress_var.set(error_num)
                error_num += 1
            if finished_threads == num_threads:
                break

        save_file_content += ''.join(content_list)

        if not self.window.stop:
            save_file = open(self.window.filename, 'w')
            save_file.write(save_file_content)
            save_file.close()
            self.msg_queue.put(0)
        else:
            self.msg_queue.put(-1)


class PostEditChunk(threading.Thread):

    def __init__(self, blast_reader, emb_en, emb_pt, chunk, queue_in, queue_out):
        threading.Thread.__init__(self)
        self.blast_reader = blast_reader
        self.emb_en = emb_en
        self.emb_pt = emb_pt
        self.chunk = chunk
        self.queue_in = queue_in
        self.queue_out = queue_out

        self.start()

    def run(self):
        for (error_index, error) in self.chunk:
            try:
                msg = self.queue_in.get_nowait()
                if msg == -1:
                    break
            except queue.Empty:
                line = error[0]
                write_line = ''
                write_line += ' '.join(self.blast_reader.src_lines[line])
                write_line += '\n'
                write_line += ' '.join(self.blast_reader.ref_lines[line])
                write_line += '\n'
                write_line += ' '.join(self.blast_reader.sys_lines[line])
                write_line += '\n'

                error_info = [','.join(map(str, e)) for e in error[1][:-1]]
                error_info.append(error[1][-1])
                write_line += '#'.join(error_info)
                write_line += '\n'

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
                write_line += '#@'.join(candidates)
                write_line += '\n'

                self.queue_out.put((error_index, write_line))
        self.queue_out.put(0)
