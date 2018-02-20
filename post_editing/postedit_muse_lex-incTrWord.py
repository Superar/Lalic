# -*- coding: utf-8 -*-
import sys
import uuid
import tkinter as tk
import tkinter.messagebox as msgb
import tkinter.filedialog as fdialog
sys.path.insert(0, '/home/marciolima/Documentos/Lalic/word_embeddings')
from read_blast import BlastReader
from read_muse_embeds import load_embeddings, closest_words

BLAST_FILE_PATH = '/home/marciolima/Dropbox/Marcio/Tarefas/1.Anotacao_erros_corpus_NMT/Blast/Entrada_Blast/test-a/FAPESP_NMT_test-a_truecased.txt'
MUSE_EN_FILE_PATH = '/home/marciolima/MUSE/dumped/gq0uc5nw4m/vectors-en.txt'
MUSE_PT_FILE_PATH = '/home/marciolima/MUSE/dumped/gq0uc5nw4m/vectors-pt.txt'


class Application(object):
    def __init__(self, master=None):
        master.title('Pós-edição automática')

        # Menu
        self.menubar = tk.Menu(master)
        self.apemenu = tk.Menu(self.menubar, tearoff=0)
        self.apemenu.add_command(label='Abrir', command=self.load_ape_file)
        self.menubar.add_cascade(label='APE', menu=self.apemenu)
        self.blastmenu = tk.Menu(self.menubar, tearoff=0)
        self.blastmenu.add_command(
            label='Abrir', command=lambda: self.load_blast_file(master))
        self.menubar.add_cascade(label='BLAST', menu=self.blastmenu)
        master.config(menu=self.menubar)

        # Src
        self.widget_src = tk.Frame(master)
        self.widget_src.grid(row=0, column=0, pady=10, padx=10)
        self.label_src = tk.Label(self.widget_src, text='Src')
        self.label_src.grid(row=0, column=0, padx=(0, 10))
        self.src_text = tk.Text(self.widget_src, height=5)
        self.src_text.insert('end', 'dedo no cu e gritaria')
        self.src_text.config(state=tk.DISABLED)
        self.src_text.grid(row=0, column=1)

        # Ref
        self.widget_ref = tk.Frame(master)
        self.widget_ref.grid(row=1, column=0, pady=10, padx=10)
        self.label_ref = tk.Label(self.widget_ref, text='Ref')
        self.label_ref.grid(row=0, column=0, padx=(0, 10))
        self.ref_text = tk.Text(self.widget_ref, height=5)
        self.ref_text.insert('end', 'dedo no cu e gritaria')
        self.ref_text.config(state=tk.DISABLED)
        self.ref_text.grid(row=0, column=1)

        # Sys
        self.widget_sys = tk.Frame(master)
        self.widget_sys.grid(row=2, column=0, pady=10, padx=10)
        self.label_sys = tk.Label(self.widget_sys, text='Sys')
        self.label_sys.grid(row=0, column=0, padx=(0, 10))
        self.sys_text = tk.Text(self.widget_sys, height=5)
        self.sys_text.insert('end', 'dedo no cu e gritaria')
        self.sys_text.config(state=tk.DISABLED)
        self.sys_text.grid(row=0, column=1)

        # APE
        self.widget_cor = tk.Frame(master)
        self.widget_cor.grid(row=3, column=0, pady=10, padx=10, sticky=tk.W)
        self.label_cor = tk.Label(self.widget_cor, text='APE')
        self.label_cor.grid(row=0, column=0, rowspan=3, padx=(0, 10))
        self.cor_text = tk.Text(self.widget_cor, width=35, height=5)
        self.cor_text.insert('end', 'dedo no cu e gritaria')
        self.cor_text.config(state=tk.DISABLED)
        self.cor_text.grid(row=0, column=1, rowspan=3, padx=(0, 10))
        self.correto_button = tk.Button(
            self.widget_cor, text='Correto', width=15)
        self.correto_button.grid(row=0, column=2)
        self.parcial_button = tk.Button(
            self.widget_cor, text='Parcialmente correto', width=15)
        self.parcial_button.grid(row=1, column=2)
        self.errado_button = tk.Button(
            self.widget_cor, text='Errado', width=15)
        self.errado_button.grid(row=2, column=2)

    def load_ape_file(self):
        fname = fdialog.askopenfilename(title='Selecione um arquivo')
        assert fname
        print(fname)
        return

    def get_filename_callback(self, event):
        filename = fdialog.askopenfile(title='Selecione um arquivo')
        assert filename

        if event.widget.message == 'BLAST':
            self.blast_path_text.config(state=tk.NORMAL)
            self.blast_path_text.delete('1.0', tk.END)
            self.blast_path_text.insert('end', filename.name)
            self.blast_path_text.config(state=tk.DISABLED)
        elif event.widget.message == 'EN':
            self.en_path_text.config(state=tk.NORMAL)
            self.en_path_text.delete('1.0', tk.END)
            self.en_path_text.insert('end', filename.name)
            self.en_path_text.config(state=tk.DISABLED)
        else:
            self.pt_path_text.config(state=tk.NORMAL)
            self.pt_path_text.delete('1.0', tk.END)
            self.pt_path_text.insert('end', filename.name)
            self.pt_path_text.config(state=tk.DISABLED)

    def load_blast(self, master):
        blast_path = self.blast_path_text.get('1.0', tk.END).strip()
        en_path = self.en_path_text.get('1.0', tk.END).strip()
        pt_path = self.pt_path_text.get('1.0', tk.END).strip()
        assert blast_path
        assert en_path
        assert pt_path

        blast_reader = BlastReader(blast_path)
        errors = blast_reader.get_filtered_errors(['lex-incTrWord'])
        emb_en, emb_pt = load_embeddings(en_path, pt_path)

        filename = str(uuid.uuid4())
        save_file = open(filename, 'w')
        save_file.write('@annotations\n')

        for error in errors:
            line = error[0]
            save_file.write(' '.join(blast_reader.src_lines[line]))
            save_file.write('\n')
            save_file.write(' '.join(blast_reader.ref_lines[line]))
            save_file.write('\n')
            save_file.write(' '.join(blast_reader.sys_lines[line]))
            save_file.write('\n')

            error_info = [','.join(map(str, e)) for e in error[1][:-1]]
            error_info.append(error[1][-1])
            save_file.write('#'.join(error_info))
            save_file.write('\n')

            sentence_to_correct = blast_reader.src_lines[line]
            for i in error[1][0]:
                if i > 0:
                    candidates = [w[0] for w in closest_words(
                        sentence_to_correct[i], emb_en, emb_pt)]
                    save_file.write('#'.join(candidates))
            save_file.write('\n')

        save_file.close()
        msgb.showinfo('Salvo', 'Arquivo salvo em: ' + filename)
        master.destroy()

    def load_blast_file(self, master):
        blast_window = tk.Toplevel(master)
        blast_widget = tk.Frame(blast_window)
        blast_widget.grid(row=0, column=0, pady=10, padx=10)

        # BLAST
        blast_path_label = tk.Label(blast_widget, text='Arquivo BLAST')
        blast_path_label.grid(row=0, column=0, sticky=tk.W)
        self.blast_path_text = tk.Text(blast_widget, height=1)
        self.blast_path_text.config(state=tk.DISABLED)
        self.blast_path_text.grid(row=0, column=1, padx=10)
        blast_path_button = tk.Button(blast_widget, text='Selecionar')
        blast_path_button.bind('<Button-1>', self.get_filename_callback)
        blast_path_button.message = 'BLAST'
        blast_path_button.grid(row=0, column=2)

        # EN
        en_path_label = tk.Label(blast_widget, text='Embeddings EN')
        en_path_label.grid(row=1, column=0, sticky=tk.W)
        self.en_path_text = tk.Text(blast_widget, height=1)
        self.en_path_text.config(state=tk.DISABLED)
        self.en_path_text.grid(row=1, column=1, padx=10)
        en_path_button = tk.Button(blast_widget, text='Selecionar')
        en_path_button.bind('<Button-1>', self.get_filename_callback)
        en_path_button.message = 'EN'
        en_path_button.grid(row=1, column=2)

        # PT
        pt_path_label = tk.Label(blast_widget, text='Embeddings PT')
        pt_path_label.grid(row=2, column=0, sticky=tk.W)
        self.pt_path_text = tk.Text(blast_widget, height=1)
        self.pt_path_text.config(state=tk.DISABLED)
        self.pt_path_text.grid(row=2, column=1, padx=10)
        pt_path_button = tk.Button(blast_widget, text='Selecionar')
        pt_path_button.bind('<Button-1>', self.get_filename_callback)
        pt_path_button.message = 'PT'
        pt_path_button.grid(row=2, column=2)

        # Concluido
        concluido_button = tk.Button(
            blast_widget, text='Concluído', command=lambda: self.load_blast(blast_window))
        concluido_button.grid(row=3, column=0, columnspan=3, pady=10)

        return

    def corrige(self):
        blast_reader = BlastReader(BLAST_FILE_PATH)
        errors = blast_reader.get_filtered_errors(['lex-incTrWord'])

        emb_en, emb_pt = load_embeddings(MUSE_EN_FILE_PATH, MUSE_PT_FILE_PATH)

        for error in errors[:2]:
            line = error[0]
            print(' '.join(blast_reader.src_lines[line]))
            print(' '.join(blast_reader.ref_lines[line]))
            print(' '.join(blast_reader.sys_lines[line]))

            sentence_to_correct = blast_reader.src_lines[line]
            sys_sentence = blast_reader.sys_lines[line]
            for i in error[1][0]:
                if i > 0:
                    for w in closest_words(sentence_to_correct[i], emb_en, emb_pt):
                        corrected_word = w[0]
                        print('corrected: {} > {}'.format(sentence_to_correct[i],
                                                          corrected_word))
                        print('\n')


root = tk.Tk()
Application(root)
root.mainloop()
