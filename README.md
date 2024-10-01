# Anchor-based-loc
This is the repository for Anchor-based Method for Multi-Camera Pedestrian Localization.
The repository contains:
1. the simulation and real dataset for localization
2. the source code of the experiments

# Requirements
Installation
1. Clone the project and create virtual environment
    ```
    conda create --name anchor_loc python=3.11
    conda activate anchor_loc
    ```
2. Install:
    ```
    pip install -r requirement.txt
    ```

# Run for each experiments
    
    cd run_experiment
The commands for getting each table and figure results are under `run_experiment` folder.
1. To get the table 1 result:
    ```
    bash table1.bash
    ```
    The corresponding table result is stored at `exp_result/exp1/table1.txt`


2. Table 2:
    ```
    bash table2.bash
    ```
    Table two result is at `exp_result/exp2/table2.txt`


3. Table 3 & 4:
    ```
    bash table3&4.bash
    ```
    Table three result is at `exp_result/exp3/table3.txt`

    Table four result is at `exp_result/exp3/table4.txt`.


4. Figure 4:
    ```
    bash figure4.bash
    ```
    The result at `exp_result/exp4/Fig4.pdf`


5. Figure 5:
    ```
    bash figure5.bash
    ```
    The result at `exp_result/exp5/Fig5.pdf`


6. Table 5:
   ```
   bash table5.bash
   ```
   Table five is at `exp_result/exp6/table5.csv`
   