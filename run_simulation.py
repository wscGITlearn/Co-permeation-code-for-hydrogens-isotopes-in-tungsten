from copermeation import build_full_simulation_params, CoPermeationSimulator
import numpy as np

def calculate_area_from_mass_and_thickness(mass_g, thickness_mm):
    """
    根据质量（克）和厚度（毫米）计算钨薄片面积（平方米）
    
    参数:
        mass_g: 质量，单位为克 (g)
        thickness_mm: 厚度，单位为毫米 (mm)
    
    返回:
        面积，单位为平方米 (m^2)
    """
    # 钨的密度 (kg/m^3)
    rho = 19250

    # 单位换算
    mass_kg = mass_g / 1000  # 克转千克
    thickness_m = thickness_mm / 1000  # 毫米转米

    # 面积计算
    area_m2 = mass_kg / (rho * thickness_m)

    return area_m2

def run_simulation(P_D2=1e5, temperature=973, sample_key = "H2:D8", isotope_input='D', k_d0=None, E_kd=None, k_r0=None, E_kr=None, material='W', total_time=2000, Delta_t=0.01):
    """
    该函数将初始化并运行 CoPermeationSimulator，进行气体渗透模拟，并计算相应的渗透流量和表面积。
    
    参数:
        P_D2 (float): D2的偏压（Pa）
        temperature (float): 温度（K）
        isotope_input (str): 氢同位素（'D' 或 'H'）
        k_d0 (float): 解俘获前因子（mol/(m²·s·Pa)）
        E_kd (float): 解俘获能（kJ/mol）
        k_r0 (float): 俘获前因子（m/s）
        E_kr (float): 俘获能（kJ/mol）
        material (str): 材料类型（'W' 或 'generic'）
        total_time (float): 总模拟时间（秒）
        Delta_t (float): 时间步长（秒）
    
    返回:
        c_H_final, c_D_final, J_H2, J_D2, J_HD (模拟结果)
    """
    # 初始化 CoPermeationSimulator 对象，使用 build_full_simulation_params 获取参数
    params = build_full_simulation_params(P_D2)
    param_set = params[sample_key]
    param_set_const = params["H5:D5"]

    # 使用提取的参数初始化 CoPermeationSimulator
    simulator = CoPermeationSimulator(
        H2_pressure=param_set["H2_pressure"], 
        D2_pressure=param_set["D2_pressure"],
        temperature=temperature, 
        gas_phase_equilibrium=False,
        isotope_input=isotope_input,
        D_0=param_set_const["D_0"],  # m²/s
        E_D=param_set_const["E_D"],  # kJ/mol
        k_d0=k_d0, 
        E_kd=E_kd,  # mol/(m²·s·Pa), kJ/mol
        k_r0=k_r0, 
        E_kr=E_kr,  # m⁴/(s·mol), kJ/mol
        K_S0=param_set_const["K_S0"], 
        E_KS=param_set_const["E_KS"],  # mol/(m³·Pa^0.5), kJ/mol
        material=material,  # 'W'
        L=param_set_const["L"]/1000,  # m
        total_time=total_time,  # s
        Delta_t=Delta_t
    )

    # 运行模拟并获取结果
    c_H_final, c_D_final, J_H2, J_D2, J_HD = simulator.simulate()
    J_D = 2*J_D2 + J_HD # mol/m^2/s

    print("simulation is complete")

    area_m2 = calculate_area_from_mass_and_thickness(param_set_const["mass"], param_set_const["L"])

    print(f"the area of Tungsten is {area_m2} m2")
    L_D2_mol = J_D2 * area_m2
    L_HD_mol = J_HD * area_m2

    # 理想气体状态方程：R=PV/nT =8.314 pa*m^3/mol/K
    # PV = nRT = 8.314*T mol*K*(pa*m^3/mol/K)
    L_D2 = L_D2_mol * 8.314 * temperature
    L_HD = L_HD_mol * 8.314 * temperature

    Ion_D = (L_D2) / 9461.953
    Ion_HD = (L_HD) / 9461.953

    time_array = np.linspace(0, (simulator.M + 1) * simulator.Delta_t, simulator.M + 1)
    space_array = np.linspace(0, simulator.N * simulator.Delta_l, simulator.N)

    return Ion_D, Ion_HD, c_H_final, c_D_final, time_array, space_array, J_D, param_set["H2_pressure"]


