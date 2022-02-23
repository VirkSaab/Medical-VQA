import pandas as pd
from typing import Union, Dict
from mvqag.utils import load_yaml
from collections import OrderedDict

__all__ = [
    "generate_new_questions_dataframe",
    "make_run_name",
]


def match_task(question: str, keywords: list):
    """Match question words with the keywords.
    Return True and the matched word if at least one word is matched.
    """
    for word in question.split(" "):
        for kw in keywords:
            if word == kw:
                return word
    return ""


def expand_abnormality_questions(
    answer: str, matched_word: str, modality: Union[None, str]
):
    """Create new questions for abnormality with given information.

    Args:
        answer: Original answer to the question
        matched_word: The keyword which labeled the original question as abnormality
        modality: name of the scan type present in original question.

    Returns:
        dict of generated questions including the original question
    """
    binary, categorical = {}, {}
    abnormal_keywords = [
        "abnormal",
        "abnormality",
        "abnormalities",
        "alarming",
        "wrong",
    ]

    if modality == "ct":
        modality = "ct scan"
    if modality == "pet":
        modality = "pet scan"

    if modality != None:
        if answer in ["yes", "no"]:
            if (answer == "no") and (matched_word not in abnormal_keywords):
                flag = "abnormal"
            elif (answer == "no") and (matched_word in abnormal_keywords):
                flag = "normal"
            elif (answer == "yes") and (matched_word not in abnormal_keywords):
                flag = "normal"
            elif (answer == "yes") and (matched_word in abnormal_keywords):
                flag = "abnormal"
            else:
                raise ValueError(
                    "[Error @ `expand_abnormality_questions`] Something wrong with the conditions"
                )
        else:
            flag = "abnormal"
            categorical[f"what is most alarming about this {modality}?"] = answer
            categorical[f"what is most alarming about the {modality}?"] = answer
            categorical[f"what is abnormal in this {modality}?"] = answer
            categorical[f"what is abnormal in the {modality}?"] = answer
            categorical[f"what abnormality is seen in this {modality}?"] = answer
            categorical[f"what abnormality is seen in the {modality}?"] = answer
            categorical[f"what is the primary abnormality in this {modality}?"] = answer
            categorical[f"what is the primary abnormality in the {modality}?"] = answer

        binary[f"is this a normal {modality}?"] = "yes" if flag == "normal" else "no"
        binary[f"is the {modality} normal?"] = "yes" if flag == "normal" else "no"
        binary[f"is this {modality} normal?"] = "yes" if flag == "normal" else "no"
        binary[f"does this {modality} look normal?"] = (
            "yes" if flag == "normal" else "no"
        )
        binary[f"are there abnormalities in this {modality}?"] = (
            "no" if flag == "normal" else "yes"
        )
        binary[f"is there an abnormality in the {modality}?"] = (
            "no" if flag == "normal" else "yes"
        )
        binary[f"is there evidence of any abnormalities?"] = (
            "no" if flag == "normal" else "yes"
        )
        binary[f"is there something wrong in the {modality}?"] = (
            "no" if flag == "normal" else "yes"
        )
    else:
        if answer in ["yes", "no"]:
            if (answer == "no") and (matched_word not in abnormal_keywords):
                flag = "abnormal"
            elif (answer == "no") and (matched_word in abnormal_keywords):
                flag = "normal"
            elif (answer == "yes") and (matched_word not in abnormal_keywords):
                flag = "normal"
            elif (answer == "yes") and (matched_word in abnormal_keywords):
                flag = "abnormal"
            else:
                raise ValueError(
                    "[Error @ `expand_abnormality_questions`] Something wrong with the conditions"
                )
        else:
            flag = "abnormal"
            categorical[f"what is most alarming about this image?"] = answer
            categorical[f"what is most alarming about the image?"] = answer
            categorical[f"what is abnormal in this image?"] = answer
            categorical[f"what is abnormal in the image?"] = answer
            categorical[f"what abnormality is seen in this image?"] = answer
            categorical[f"what abnormality is seen in the image?"] = answer
            categorical[f"what is the primary abnormality in this image?"] = answer
            categorical[f"what is the primary abnormality in the image?"] = answer

        binary[f"is this a normal image?"] = "yes" if flag == "normal" else "no"
        binary[f"is the image normal?"] = "yes" if flag == "normal" else "no"
        binary[f"is this image normal?"] = "yes" if flag == "normal" else "no"
        binary[f"does this image look normal?"] = "yes" if flag == "normal" else "no"
        binary[f"are there abnormalities in this image?"] = (
            "no" if flag == "normal" else "yes"
        )
        binary[f"is there an abnormality in the image?"] = (
            "no" if flag == "normal" else "yes"
        )
        binary[f"is there evidence of any abnormalities?"] = (
            "no" if flag == "normal" else "yes"
        )
        binary[f"is there something wrong in the image?"] = (
            "no" if flag == "normal" else "yes"
        )
    return {"binary": binary, "categorical": categorical}


def expand_modality_questions(answer: str, matched_word: str):
    """Create new questions for modality task with given information.

    Args:
        answer: Original answer to the question
        matched_word: The keyword which labeled the original question as modality

    Returns:
        dict of generated questions including the original question
    """
    if matched_word not in ["modality", "kind"]:
        if matched_word == "ct":
            matched_word = "ct scan"
        if matched_word == "pet":
            matched_word = "pet scan"
        answer = matched_word

    categorical = {}
    categorical["in what modality is this image taken?"] = answer
    categorical["which image modality is this?"] = answer
    categorical["with what modality is this image taken?"] = answer
    categorical["what is the modality?"] = answer
    categorical["what is the modality of this image?"] = answer
    categorical["what modality is shown?"] = answer
    categorical["what modality is used to take this image?"] = answer
    categorical["what modality was used to take this image?"] = answer
    categorical["what imaging modality is used?"] = answer
    categorical["what imaging modality was used?"] = answer
    categorical["what imaging modality is seen here?"] = answer
    categorical["what imaging modality was used to take this image?"] = answer
    categorical["what imaging modality is used to acquire this picture?"] = answer
    categorical["what type of image modality is this?"] = answer
    categorical["what type of imaging modality is shown?"] = answer
    categorical["what type of imaging modality is seen in this image?"] = answer
    categorical["what type of image modality is seen?"] = answer
    categorical["what type of imaging modality is used to acquire the image?"] = answer
    categorical["what kind of image is this?"] = answer
    categorical["how is the image taken?"] = answer
    categorical["how was the image taken?"] = answer
    categorical["how was the image taken?"] = answer
    categorical["how was this image taken?"] = answer
    return {"binary": {}, "categorical": categorical}


def expand_fine_modality_questions(
    answer: str, matched_word: str, modality: Union[None, str]
):
    """Create new questions for fine modality task with given information.

    Args:
        answer: Original answer to the question
        matched_word: The keyword which labeled the original question as fine_modality
        modality: name of the scan type present in original question.

    Returns:
        dict of generated questions including the original question
    """
    binary, categorical = {}, {}

    if modality == "ct":
        modality = "ct scan"
    if modality == "pet":
        modality = "pet scan"

    if answer in ["yes", "no"]:
        if matched_word == "iv_contrast":
            binary["was iv_contrast given to the patient?"] = answer
        elif matched_word == "gi_contrast":
            binary["was gi_contrast given to the patient?"] = answer

        if ("t1" in matched_word) and answer == "yes":
            binary["is this a t1_weighted image?"] = "yes"
            binary["is this a t2_weighted image?"] = "no"
            binary["is this a flair image?"] = "no"
        if ("t1" in matched_word) and answer == "no":
            binary["is this a t1_weighted image?"] = "no"

        if ("t2" in matched_word) and answer == "yes":
            binary["is this a t1_weighted image?"] = "no"
            binary["is this a t2_weighted image?"] = "yes"
            binary["is this a flair image?"] = "no"
        if ("t2" in matched_word) and answer == "no":
            binary["is this a t2_weighted image?"] = "no"

        if ("flair" in matched_word) and answer == "yes":
            binary["is this a t1_weighted image?"] = "no"
            binary["is this a t2_weighted image?"] = "no"
            binary["is this a flair image?"] = "yes"
        if ("flair" in matched_word) and answer == "no":
            binary["is this a flair image?"] = "no"

        if (matched_word == "contrast") and modality:
            binary[f"is this a noncontrast {modality}?"] = (
                "no" if answer == "yes" else "yes"
            )
            binary[f"was the {modality} taken with contrast?"] = (
                "yes" if answer == "yes" else "no"
            )
        if (matched_word == "noncontrast") and modality:
            binary[f"is this a noncontrast {modality}?"] = (
                "yes" if answer == "yes" else "no"
            )
            binary[f"was the {modality} taken with contrast?"] = (
                "no" if answer == "yes" else "yes"
            )
    else:
        if matched_word == "contrast":
            categorical["what type of contrast did this patient have?"] = answer
        if ("t1" in answer) or ("t2" in answer) or ("flair" in answer):
            categorical["is this a t1_weighted, t2_weighted, or flair image?"] = answer
            categorical["is this image modality t1, t2, or flair?"] = answer
            if "t1" in answer:
                binary["is this a t1_weighted image?"] = "yes"
            elif "t2" in answer:
                binary["is this a t2_weighted image?"] = "yes"
            elif "flair" in answer:
                binary["is this a flair image?"] = "yes"
        else:
            binary["is this a t1_weighted image?"] = "no"
            binary["is this a t2_weighted image?"] = "no"
            binary["is this a flair image?"] = "no"

    return {"binary": binary, "categorical": categorical}


def expand_organ_questions(answer: str, matched_word: str, modality: Union[None, str]):
    """Create new questions for organ task with given information.

    Args:
        answer: Original answer to the question
        matched_word: The keyword which labeled the original question as organ
        modality: name of the scan type present in original question.

    Returns:
        dict of generated questions including the original question
    """
    categorical = {}

    if modality == "ct":
        modality = "ct scan"
    if modality == "pet":
        modality = "pet scan"

    if modality != None:
        categorical[f"what part of the body does this {modality} show?"] = answer
        categorical[f"which organ is captured by this {modality}?"] = answer
        categorical[f"the {modality} shows what organ system?"] = answer
        categorical[f"what organ system is shown in this {modality}?"] = answer
        categorical[
            f"what organ systems can be evaluated with this {modality}?"
        ] = answer
        categorical[f"what organ is this {modality} showing?"] = answer
        categorical[f"what is the organ principally shown in this {modality}?"] = answer
        categorical[f"what organ system is displayed in this {modality}?"] = answer
        categorical[f"which organ system is shown in the {modality}?"] = answer
    else:
        categorical["what organ system is shown in this image?"] = answer
        categorical["what is the organ system in this image?"] = answer
        categorical["what is the organ principally shown in this image?"] = answer
        categorical["what part of the body is being imaged?"] = answer
        categorical["what part of the body is being imaged here?"] = answer
        categorical["what part of the body does this image show?"] = answer
        categorical["what organ system is primarily present in this image?"] = answer
        categorical["what organ system is shown in the image?"] = answer
        categorical["what is one organ system seen in this image?"] = answer
        categorical["which organ system is imaged?"] = answer
        categorical["what organ system is visualized?"] = answer
        categorical["what organ system is pictured here?"] = answer
        categorical["what organ system is evaluated primarily?"] = answer
        categorical["what organ system is imaged?"] = answer
        categorical["the image shows what organ system?"] = answer
        categorical["what organ system is being imaged?"] = answer
        categorical["what organ is this image of?"] = answer
    return {"binary": {}, "categorical": categorical}


def expand_plane_questions(answer: str, matched_word: str, modality: Union[None, str]):
    """Create new questions for plane task with given information.

    Args:
        answer: Original answer to the question
        matched_word: The keyword which labeled the original question as plane
        modality: name of the scan type present in original question.

    Returns:
        dict of generated questions including the original question
    """
    categorical = {}

    if modality == "ct":
        modality = "ct scan"
    if modality == "pet":
        modality = "pet scan"

    if modality != None:
        categorical[f"in what plane is this {modality}?"] = answer
        categorical[f"in what plane is this {modality} captured?"] = answer
        categorical[f"in what plane is this {modality} taken?"] = answer
        categorical[f"in which plane is the {modality} displayed?"] = answer
        categorical[f"what is the plane of the {modality}?"] = answer
        categorical[f"what is the plane of this {modality}?"] = answer
        categorical[f"what plane is this {modality} in?"] = answer
        categorical[f"what plane is this {modality}?"] = answer
        categorical[f"what plane was used in this {modality}?"] = answer
        categorical[f"which plane is this {modality} taken in?"] = answer
    else:
        categorical["in what plane is this image oriented?"] = answer
        categorical["in what plane is this image?"] = answer
        categorical["in what plane is this image captured?"] = answer
        categorical["in what plane is this image taken?"] = answer
        categorical["in which plane is the image displayed?"] = answer
        categorical["in what plane was this image oriented?"] = answer
        categorical["in what plane was this image taken?"] = answer
        categorical["what image plane is this?"] = answer
        categorical["what imaging plane is depicted here?"] = answer
        categorical["what is the plane?"] = answer
        categorical["what is the plane of the image?"] = answer
        categorical["what is the plane of this image?"] = answer
        categorical["what plane is this imag?"] = answer
        categorical["what plane is demonstrated?"] = answer
        categorical["what plane is seen?"] = answer
        categorical["what plane is the image acquired in?"] = answer
        categorical["what plane is this?"] = answer
        categorical["what plane was used?"] = answer
        categorical["which plane is the image shown in?"] = answer
        categorical["which plane is the image taken?"] = answer
        categorical["which plane is this image in?"] = answer
        categorical["which plane is this image taken?"] = answer
    return {"binary": {}, "categorical": categorical}


def analyse_question(question: str, answer: str, task_keywords: Dict[str, list]):
    """Extract information from the question and answer"""
    tasks = {k: None for k in task_keywords.keys()}

    # match question with tasks
    tasks = {
        _task: match_task(question, task_keywords[_task]) for _task in tasks.keys()
    }
    #     print("Tasks matched =", [t for t, v in tasks.items() if v])
    taskwise_questions = {k: {} for k in task_keywords.keys()}

    # Abnormality task
    if tasks["abnormality"]:
        if tasks["modality"]:
            taskwise_questions["abnormality"].update(
                expand_abnormality_questions(
                    answer, tasks["abnormality"], modality=tasks["modality"]
                )
            )
        else:
            taskwise_questions["abnormality"].update(
                expand_abnormality_questions(
                    answer, tasks["abnormality"], modality=None
                )
            )

    if tasks["modality"] and (answer not in ["yes", "no"]):
        output = expand_modality_questions(answer, tasks["modality"])
        if answer in task_keywords["fine_modality"]:
            taskwise_questions["fine_modality"].update(output)
        else:
            taskwise_questions["modality"].update(output)

    if tasks["fine_modality"]:
        if tasks["modality"]:
            taskwise_questions["fine_modality"].update(
                expand_fine_modality_questions(
                    answer, tasks["fine_modality"], modality=tasks["modality"]
                )
            )
        else:
            taskwise_questions["fine_modality"].update(
                expand_fine_modality_questions(
                    answer, tasks["fine_modality"], modality=None
                )
            )

    if tasks["organ"]:
        if tasks["modality"]:
            taskwise_questions["organ"].update(
                expand_organ_questions(
                    answer, tasks["organ"], modality=tasks["modality"]
                )
            )
        else:
            taskwise_questions["organ"].update(
                expand_organ_questions(answer, tasks["organ"], modality=None)
            )

    if tasks["plane"]:
        if tasks["modality"]:
            taskwise_questions["plane"].update(
                expand_organ_questions(
                    answer, tasks["plane"], modality=tasks["modality"]
                )
            )
        else:
            taskwise_questions["plane"].update(
                expand_organ_questions(answer, tasks["plane"], modality=None)
            )

    return taskwise_questions


def generate_new_questions_dataframe(df: pd.DataFrame, task_keywords: Dict[str, list]):
    # with SpinCursor("Generating questions...", end="Questions generated."):
    columns = list(df.columns) + ["Task", "SubTask"]
    newdf = []
    for row in df.itertuples():
        try:
            taskwise_questions = analyse_question(row.Q, row.A, task_keywords)
            for task_name in taskwise_questions.keys():
                if taskwise_questions[task_name]:
                    for subtask_name, qa_pairs in taskwise_questions[
                        task_name
                    ].items():
                        if qa_pairs:
                            for Q, A in qa_pairs.items():
                                sample = OrderedDict({
                                    k: v for k, v in zip(columns, row[1:])
                                })
                                sample['Q'] = Q
                                sample['A'] = A
                                sample['Task'] = task_name
                                sample['subtask'] = subtask_name
                                newdf.append(list(sample.values()))
        except:
            print(row)
            raise RuntimeError("[Error @ `generate_new_questions_dataframe`]")
    return (
        pd.DataFrame(newdf, columns=columns)
        .drop_duplicates(keep='last')
        .reset_index(drop=True)
    )


def make_run_name(
    CNF_path,
    exp_no,
    topic,
    task_name,
    inp_size,
    batch_size,
    epochs,
    lr,
    optimizer,
    vnet_name,
    tnet_name,
):
    """Create run name for wandb experiments listing.
    If the variable is different from the default value (in config/default.yaml), it will appear in the run name.
    """
    CNF0 = load_yaml(CNF_path)

    run_name = f"{topic}-{task_name}"

    if (CNF0.model.inp_size != inp_size) and (inp_size != None):
        run_name += f"-I{inp_size}"

    if (CNF0.data.batch_size != batch_size) and (batch_size != None):
        run_name += f"-B{batch_size}"

    if (CNF0.train.epochs != epochs) and (epochs != None):
        run_name += f"-E{epochs}"

    if (CNF0.train.lr != lr) and (lr != None):
        run_name += f"-L{lr}"

    if (CNF0.train.optimizer != optimizer) and (optimizer != ""):
        run_name += f"-O{optimizer.upper()[0]}"

    if (CNF0.model.vnet_name != vnet_name) and (vnet_name != ""):
        run_name += f"-VN{vnet_name}"

    if (CNF0.model.tnet_name != tnet_name) and (tnet_name != ""):
        run_name += f"-TN{tnet_name}"

    run_name += f"-{exp_no}"

    return run_name
