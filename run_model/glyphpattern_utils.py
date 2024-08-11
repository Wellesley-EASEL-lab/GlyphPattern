def glyph_pattern_argparser(parser):
    parser.add_argument("--fewshot-prompt",
        choices=["1shot","3shot", "5shot"]) # or None
    return parser 

def create_fewshot_dataset(dataset, prompt_kind):
    oneshot = ['53_color','53_leftright','53_circle']
    threeshot = ['53_color', '210_color', '298_color','53_leftright', '210_leftright','298_leftright','53_circle','210_circle','298_circle']
    fiveshot = ['53_color', '192_color', '210_color', '268_color', '298_color','53_leftright', '192_leftright', '210_leftright', '268_leftright', '298_leftright','53_circle', '192_circle', '210_circle', '268_circle', '298_circle']
    
    if prompt_kind is None:
        return
    elif prompt_kind == "1shot":
        filenames = oneshot
    elif prompt_kind == "3shot":
        filenames = threeshot
    elif prompt_kind == "5shot":
        filenames = fiveshot

    fewshot_dataset = dataset.filter(lambda example: example['file_name'] in filenames)

    return fewshot_dataset

