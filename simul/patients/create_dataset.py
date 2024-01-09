from typing import Dict, Tuple

import pandas as pd
import numpy as np

from simul.patients.dataset import Dataset
from simul.cnv.sampling import ProgramDistribution

########## Simulation ##################
def simulate_malignant_comp_batches(dataset: Dataset, prob_dist: ProgramDistribution) -> Dict[str, pd.DataFrame]:
    """Simulates the composition of mallignant cells for a dataset

    Args:

        dataset: an instantiated Dataset object
        prob_dist: an instantiated ProgramDistribution object with the conditional probabilities
            of P (program | anchors, patient)

    Returns:

        a dictionary with patient name as key and an observation dataframe as value

    """
    all_malignant_obs = {}
    for patient in dataset.patients:

        patient_subclones = [patient.subclones[i].name for i in range(len(patient.subclones))]
        patient_subclone_profiles = {
            patient.subclones[i].name: tuple(patient.subclones[i].anchor_profile) for i in range(len(patient.subclones))
        }

        # pick the subclones the cells belong to
        batch_clones = np.random.choice(
            patient_subclones,
            size=(patient.n_malignant_cells,),
            p=patient.subclone_proportions,
        )
        cell_programs = []
        for c in batch_clones:
            cell_programs.append(
                prob_dist.sample(
                    anchors=patient_subclone_profiles[c],
                    batch=patient.batch,
                    n_samples=1,
                )[0]
            )
        malignant = ["malignant"] * patient.n_malignant_cells
        df_obs = pd.DataFrame(
            np.array([batch_clones, cell_programs, malignant]),
            index=["subclone", "program", "malignant_key"],
            columns=[f"cell{i+1}" for i in range(batch_clones.shape[0])],
        ).T
        print(df_obs)
        all_malignant_obs[patient.batch] = df_obs
    return all_malignant_obs


def drop_rarest_program(
    all_malignant_obs: Dict[str, pd.Series],
    dataset: Dataset,
    p_1: float = 0.3,
    p_2: float = 0.5,
) -> Tuple[Dict[str, pd.Series], Dataset]:

    mapping_patients = dataset.name_to_patient()
    new_obs = {}

    for patient in all_malignant_obs:
        new_obs[patient] = all_malignant_obs[patient].copy()
        sort_programs = new_obs[patient].program.value_counts().sort_values()
        rarest_program = sort_programs.index[0]
        second_rarest_program = sort_programs.index[1]
        if np.random.binomial(p=p_1, n=1, size=1):
            # drop the rarest progam
            new_obs[patient] = new_obs[patient][~(new_obs[patient].program == rarest_program)]
            # update the number of malignant cells
            mapping_patients[patient].n_malignant_cells = new_obs[patient].shape[0]

            # only if the rarest program was dropped is there the possibility for
            # the second rarest to be dropped
            if np.random.binomial(p=p_2, n=1, size=1):
                # drop the second rarest progam
                new_obs[patient] = new_obs[patient][~(new_obs[patient].program == second_rarest_program)]
                # update the number of malignant cells
                mapping_patients[patient].n_malignant_cells = new_obs[patient].shape[0]
    return new_obs, dataset


def simulate_healthy_comp_batches(dataset: Dataset, cell_types_avail: list[str]=[]) -> Dict[str, pd.DataFrame]:
    """Simulates the composition of healthy cells for a dataset

    Args:

        dataset: an instantiated Dataset object

    Returns:

        a dictionary with patient name as key and an observation dataframe as value

    """
    all_healthy_obs = {}
    for patient in dataset.patients:
        clones = ["NA"] * patient.n_healthy_cells
        cell_programs = [np.random.choice(cell_types_avail) for _ in range(patient.n_healthy_cells)]
        malignant = ["non_malignant"] * patient.n_healthy_cells

        df_obs = pd.DataFrame(
            np.array([clones, cell_programs, malignant]),
            index=["subclone", "program", "malignant_key"],
            columns=[f"cell{i+1+patient.n_malignant_cells}" for i in range(len(clones))],
        ).T
        all_healthy_obs[patient.batch] = df_obs

    return all_healthy_obs


def get_full_obs(
    all_malignant_obs: Dict[str, pd.DataFrame], all_healthy_obs: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:

    full_obs = {pat: pd.concat([all_malignant_obs[pat], all_healthy_obs[pat]]) for pat in all_malignant_obs}
    for pat in full_obs:
        full_obs[pat].index = [f"cell{i+1}" for i in range(full_obs[pat].shape[0])]
    return full_obs
