"""model_2_4.py

Flexible version of the life-cycle model in model_2_1.py.

The function simulate() reproduces the baseline simulation exactly when it is
called with its default arguments, but it also allows the individual
mechanisms of the model to be switched off one at a time (question 2.4) and
adds an extra source of risk (question 2.5).

The order of the random draws is identical to the one in model_2_1.py, so the
alternative simulations differ from the baseline only because of the
mechanism that is switched off, and not because of different random numbers.
"""

import numpy as np

# a. population
N = 50_000
ages = np.arange(18, 66)

# b. education parameters
educations = np.array(['short', 'medium', 'long'])
p_e = np.array([0.40, 0.35, 0.25])
s_e = np.array([1, 3, 5])

# c. human capital parameters
h_e0 = np.array([1.00, 1.20, 1.55])
Delta_e = np.array([0.010, 0.020, 0.030])
delta = 0.06
sigma_psi = 0.10

# d. labor market parameters
job_finding = 0.60
job_separation = 0.05

# e. income parameters
student_grant = 0.45
replacement_rate = 0.60
benefit_floor = 0.35


def simulate(educational_differences=True,
             shocks=True,
             depreciation=True,
             unemployment=True,
             common_education=1,
             p_health=0.0,
             job_separation_alt=None,
             seed=123):
    """ simulate the life-cycle model

    Args:
        educational_differences (bool): if False, all individuals get the
            same education type, so they have the same length of education,
            the same initial human capital and the same growth rate
        shocks (bool): if False, the human capital shock is switched off
            (sigma_psi = 0, so psi = 1 for everybody)
        depreciation (bool): if False, human capital does not depreciate
            while unemployed (delta = 0)
        unemployment (bool): if False, nobody is ever unemployed after the
            year of entry (job_finding = 1 and job_separation = 0)
        common_education (int): education type assigned to everybody when
            educational_differences is False (0 = short, 1 = medium,
            2 = long)
        p_health (float): annual probability of a permanent health shock
            (question 2.5). Zero in the baseline model.
        seed (int): seed of the random number generator

    Returns:
        (dict): dictionary with the simulated variables
    """

    # a. switch mechanisms off by adjusting the parameters
    s_e_ = s_e.copy()
    h_e0_ = h_e0.copy()
    Delta_e_ = Delta_e.copy()

    if not educational_differences:
        s_e_[:] = s_e[common_education]
        h_e0_[:] = h_e0[common_education]
        Delta_e_[:] = Delta_e[common_education]

    sigma_psi_ = sigma_psi if shocks else 0.0
    delta_ = delta if depreciation else 0.0
    job_finding_ = job_finding if unemployment else 1.0
    job_separation_ = job_separation if unemployment else 0.0

    # optional override of the job-separation probability, used in question
    # 2.5 to compare permanent and temporary non-employment risk
    if job_separation_alt is not None:
        job_separation_ = job_separation_alt

    # b. random number generators
    rng = np.random.default_rng(seed=seed)

    # separate generator for the health shock, so that the draws used by the
    # baseline model are not affected by the extension in question 2.5
    rng_health = np.random.default_rng(seed=seed + 1)

    # c. draw education and assign education-specific values
    education_i = rng.choice(3, size=N, p=p_e)

    education = educations[education_i]
    education_years = s_e_[education_i]
    entry_age = 18 + education_years

    initial_h = h_e0_[education_i]
    growth_rate = Delta_e_[education_i]

    # d. income while in education
    in_education = ages[None, :] < entry_age[:, None]

    income = np.zeros((N, len(ages)))
    income[in_education] = student_grant

    # e. simulate employment transitions over time
    employed = np.zeros((N, len(ages)), dtype=bool)

    # keep track of individuals who have been hit by a health shock
    disabled = np.zeros((N, len(ages)), dtype=bool)
    is_disabled = np.zeros(N, dtype=bool)

    for t in range(1, len(ages)):
        age = ages[t]

        in_labor_market = age >= entry_age
        just_entered = age == entry_age

        was_employed = employed[:, t-1]
        was_unemployed = ~was_employed

        find_job = rng.random(N) < job_finding_
        lose_job = rng.random(N) < job_separation_

        # the health shock is permanent and only hits individuals who have
        # entered the labor market and are not already disabled
        if p_health > 0:
            new_shock = (
                in_labor_market
                & ~is_disabled
                & (rng_health.random(N) < p_health)
            )
            is_disabled = is_disabled | new_shock

        disabled[:, t] = is_disabled

        # disabled individuals never work again
        employed[:, t] = (
            in_labor_market
            & ~just_entered
            & ~is_disabled
            & (
                (was_unemployed & find_job)
                | (was_employed & ~lose_job)
            )
        )

    # f. simulate human capital
    human_capital = np.zeros((N, len(ages)))
    human_capital[:, 0] = initial_h

    for t in range(1, len(ages)):

        previous_h = human_capital[:, t-1]

        psi = rng.lognormal(
            -0.5 * sigma_psi_**2,
            sigma_psi_,
            size=N
        )

        age = ages[t]

        is_studying = age < entry_age
        just_entered = age == entry_age
        is_employed = (age > entry_age) & employed[:, t]
        is_unemployed = (age > entry_age) & ~employed[:, t]

        # unchanged while studying
        human_capital[is_studying, t] = initial_h[is_studying]

        # enter labor market with education-specific initial human capital
        human_capital[just_entered, t] = initial_h[just_entered]

        # employed
        human_capital[is_employed, t] = (
            previous_h[is_employed]
            * (1 + growth_rate[is_employed])
            * psi[is_employed]
        )

        # unemployed
        human_capital[is_unemployed, t] = (
            previous_h[is_unemployed]
            * (1 - delta_)
            * psi[is_unemployed]
        )

    # g. simulate income
    last_job_income = np.zeros(N)

    for t, age in enumerate(ages):

        is_employed = employed[:, t]

        # employed individuals earn their human capital
        income[is_employed, t] = human_capital[is_employed, t]

        # remember their most recent job income
        last_job_income[is_employed] = income[is_employed, t]

        in_labor_market = age >= entry_age

        is_unemployed = in_labor_market & ~is_employed
        has_worked_before = last_job_income > 0

        # benefit for individuals who have worked before
        income[is_unemployed & has_worked_before, t] = (
            replacement_rate
            * last_job_income[is_unemployed & has_worked_before]
        )

        # benefit floor for individuals who have never worked before
        income[is_unemployed & ~has_worked_before, t] = benefit_floor

    return {
        'income': income,
        'human_capital': human_capital,
        'employed': employed,
        'disabled': disabled,
        'education': education,
        'entry_age': entry_age,
        'ages': ages,
    }
