
# a. build model
import numpy as np

# population
N = 50_000
ages = np.arange(18, 65)

# b. build random generator for educations with propabilities p

# education parameters
educations = np.array(['short', 'medium', 'long'])
p_e = np.array([0.40, 0.35, 0.25])
s_e = np.array([1, 3, 5])

# random generator
rng = np.random.default_rng(seed=123)

# draw education type for each individual
education_i = rng.choice(3, size=N, p=p_e)

# c. assign education type and education length to each individual

education = educations[education_i]

education_years = s_e[education_i]

# income parameters for students
student_grant = 0.45
entry_age = 18 + education_years 

# identify years spent in education
in_education = ages[None, :] < entry_age[:, None]

# initialize annual income
income = np.zeros((N, len(ages)))

# applying income to be equal to 0.45 in the matrix, where in_education is True
income[in_education] = student_grant

# d. parameters for labor market
job_finding = 0.6
job_separation = 0.05

# initialize employment status 
employed = np.zeros((N, len(ages)), dtype=bool)

# simulate employment transitions over time

for t in range(1, len(ages)):
    age = ages[t]

    in_labor_market = age >= entry_age
    just_entered = age == entry_age

    was_employed = employed[:, t-1]
    was_unemployed = ~was_employed

    find_job = rng.random(N) < job_finding
    lose_job = rng.random(N) < job_separation

    employed[:, t] = (
        in_labor_market
        & ~just_entered
        & (
            (was_unemployed & find_job)
            | (was_employed & ~lose_job)
        )
    )


# e. building up the human capital by using the employment status 

# human capital parameters 

h_e0 = np.array([1.00, 1.20, 1.55]) # Initial human capital
Delta_e = np.array([0.010, 0.020, 0.030]) # Growth of human capital

delta = 0.06 # Depreciation while unemployed
sigma_psi = 0.10 # Std. of shock

# education-specific values for each individual
initial_h = h_e0[education_i]
growth_rate = Delta_e[education_i]

# human capital matrix 
human_capital = np.zeros((N, len(ages)))

# human capital at age 18
human_capital[:, 0] = initial_h

# human capital loop
for t in range(1, len(ages)):

    previous_h = human_capital[:, t-1]

    psi = rng.lognormal(
        -0.5 * sigma_psi**2,
        sigma_psi,
        size=N
    )

    age = ages[t]

    is_studying = age < entry_age
    just_entered = age == entry_age
    was_employed = employed[:, t-1]
    was_unemployed = ~was_employed

    is_employed = (age > entry_age) & was_employed
    is_unemployed = (age > entry_age) & was_unemployed

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
        * (1 - delta)
        * psi[is_unemployed]
    )

# f. adding income to the simulation 

# income parameters
replacement_rate = 0.60
benefit_floor = 0.35

# keep track of income in the most recent job
last_job_income = np.zeros(N)

# calculate income over the life cycle
for t, age in enumerate(ages):

    # identify employed individuals
    is_employed = employed[:, t]

    # employed individuals earn their human capital
    income[is_employed, t] = human_capital[is_employed, t]

    # remember their most recent job income
    last_job_income[is_employed] = income[is_employed, t]

    # identify individuals who have entered the labor market
    in_labor_market = age >= entry_age

    # identify unemployed individuals
    is_unemployed = in_labor_market & ~is_employed

    # identify unemployed individuals who have worked before
    has_worked_before = last_job_income > 0

    # unemployment benefit for individuals who have worked before
    income[is_unemployed & has_worked_before, t] = (
        replacement_rate
        * last_job_income[is_unemployed & has_worked_before]
    )

    # benefit floor for individuals who have never worked before
    income[is_unemployed & ~has_worked_before, t] = benefit_floor

