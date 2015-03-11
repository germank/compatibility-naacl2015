#!/usr/bin/env python
import os
import glob


def main():
    model_files = glob.glob("../models/l*.lua")
    model_names = []
    for model_file in model_files:
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        model_name = model_name.replace("-", "_")
        model_names.append(model_name)

    objs_combs = [(True,True)]# [(True,True),(True,False),(False,True)]
    use_cosine = ["false"]
    regs = ["1e-1", "1e-2","1e-3","1e-4","1e-5"]

    index = 1
    for model_name in model_names:
        for reg in regs:
            for uc in use_cosine:
                for obj_comb in objs_combs:
                    gen_file(index, obj_comb, model_name, reg, uc)
                    index += 1

def gen_file(index, obj_comb, model_name, reg, use_cosine):
    with open('conf_{0:02}.lua'.format(index), 'w') as f:
        f.write("opt.coef_L2 = {0}\n".format(reg))
        f.write("opt.use_cosine = {0}\n".format(use_cosine))
        f.write("model_name = '{0}'\n".format(model_name))
        use_corr, use_class = obj_comb
        if use_corr:
            f.write("obj['corr'] = true\n")
        else:
            f.write("obj['corr'] = false\n")
        if use_class:
            f.write("obj['pos'] = true\n")
            f.write("obj['neg'] = true\n")
        else:
            f.write("obj['pos'] = false\n")
            f.write("obj['neg'] = false\n")
            


if __name__ == '__main__':
    main()
