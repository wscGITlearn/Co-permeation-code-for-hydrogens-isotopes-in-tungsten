import numpy as np

def build_full_simulation_params(P_D2=1e5):
    """
    构建包含所有实验数据的参数字典（H2压力、厚度、质量、D0、E_D、KS0、E_KS）
    所有激活能均为 kJ/mol，KS0 为 m/s，适用于 CoPermeationSimulator。
    """
    full_data = {
        "H0:D10": {"L": 0.277, "mass": 0.5493, "D_0": 2.35e-4, "E_D": 114.17, "K_S0": 1.26e-3, "E_KS": 11.16},
        "H1:D9":  {"L": 0.257, "mass": 0.5473, "D_0": 8.28e-4, "E_D": 123.09, "K_S0": 7.38e-4, "E_KS": -11.32},
        "H2:D8":  {"L": 0.289, "mass": 0.5332, "D_0": 6.37e-5, "E_D": 100.52, "K_S0": 1.65e-3, "E_KS": 15.49},
        "H3:D7":  {"L": 0.263, "mass": 0.4716, "D_0": 9.53e-6, "E_D": 81.68,  "K_S0": 9.79e-3, "E_KS": 32.90},
        "H4:D6":  {"L": 0.275, "mass": 0.4210, "D_0": 5.08e-8, "E_D": 54.63,  "K_S0": 1.32,    "E_KS": 47.13},
        "H5:D5":  {"L": 0.301, "mass": 0.6473, "D_0": 5.08e-8, "E_D": 54.63,  "K_S0": 1.32,    "E_KS": 47.13}
    }

    param_dict = {}
    for key, v in full_data.items():
        h, d = map(int, key.replace("H", "").replace("D", "").split(":"))
        P_H2 = (P_D2 * h / d) if d != 0 else 0.0
        param_dict[key] = {
            "H2_pressure": round(P_H2, 4),
            "D2_pressure": P_D2,
            "D_0": v["D_0"],
            "E_D": -v["E_D"],
            "K_S0": v["K_S0"],
            "E_KS": -v["E_KS"],
            "L": v["L"],
            "mass": v["mass"]
        }

    return param_dict

class CoPermeationSimulator:
    R = 8.314                     # J/(mol·K)
    kB_ev = 8.61732814974056e-5   # eV/K
    NA = 6.02214076e23            # 1/mol

    def __init__(self,
                 # 气压、温度、同位素等常规参数……
                 H2_pressure, D2_pressure,
                 temperature, gas_phase_equilibrium,
                 isotope_input,
                 D_0, E_D,
                 # surface 反应可选参数
                 k_d0=None, E_kd=None,
                 k_r0=None, E_kr=None,
                 # 溶解度参数，缺 kd/kr 时必须
                 K_S0=None, E_KS=None,
                 # 材料类型：'generic' 或 'W'
                 material='generic',
                 L=None, total_time=None, Delta_t=0.01):
        """
        初始化 co-permeation 模拟所有参数。

        参数：
        - H2_pressure (float): H₂偏压 (Pa)
        - D2_pressure (float): D₂偏压 (Pa)
        - temperature (float): 温度 (K)
        - gas_phase_equilibrium (bool): 是否执行气相平衡
        - isotope_input (str): 输入的参数同位素类型 ("H" 或 "D")
        - D_0, E_D, k_d0, E_kd, k_r0, E_kr, k_s0, E_ks: 材料参数
        - L (float): 材料膜厚 (m)
        - total_time (float): 模拟总时间 (s)
        - Delta_t (float, optional): 时间步长 (s)，默认0.01s
        """
        self.H2_pressure = H2_pressure
        self.D2_pressure = D2_pressure
        self.total_pressure = H2_pressure + D2_pressure # 守恒的总压强

        self.temperature = temperature
        self.gas_phase_equilibrium = gas_phase_equilibrium

        if gas_phase_equilibrium: 
            # 反应后HD气偏压，非守恒
            self.HD_pressure = self.calculate_HD_pressure()
            # 守恒后各气体偏压：
            self.pressure_sum = self.HD_pressure + self.H2_pressure + self.D2_pressure
            self.H2_pressure = self.H2_pressure * self.total_pressure / self.pressure_sum
            self.D2_pressure = self.D2_pressure * self.total_pressure / self.pressure_sum
            self.H2_pressure = self.HD_pressure * self.total_pressure / self.pressure_sum
        else :self.HD_pressure = 0

        assert isotope_input in ['H', 'D'], "isotope_input必须是'H'或'D'"
        self.isotope_input = isotope_input

        self.D_0 = D_0
        self.E_D = E_D
        self.D = self.arrhenius(self.D_0, self.E_D)

        self.L = L
        self.total_time = total_time
        self.Delta_t = Delta_t  # 初始时间步长

        self.beta_factors = {
            'H_to_D': 1 / np.sqrt(2),
            'D_to_H': np.sqrt(2)
        }

        kd_provided = (k_d0 is not None and E_kd is not None)
        kr_provided = (k_r0 is not None and E_kr is not None)

        if kd_provided and kr_provided:
            # 1) 两对都提供：直接 Arrhenius
            self.K_S = None
            self.kr_atom = None
            self.k_d0, self.E_kd = k_d0, E_kd
            self.k_r0, self.E_kr = k_r0, E_kr
            self.k_d = self.arrhenius(self.k_d0, self.E_kd)
            self.k_r = self.arrhenius(self.k_r0, self.E_kr)

        else:
            # 2) 缺失任一对，这时才需要 Ks
            assert K_S0 is not None and E_KS is not None, "缺 kd/kr 时必须提供 K_S0/E_KS"
            self.K_S0, self.E_KS = K_S0, E_KS
            self.K_S = self.arrhenius(self.K_S0, self.E_KS)
            self.kr_atom = None

            if material.lower() == 'w':
                print(f"材料是W, k_r选用Mingzhong Zhao/I. Takagi的结果。")
                # W 的专用 kr(T) → mol 单位
                T = self.temperature
                self.kr_atom = 4.5e-25 * np.exp(-0.78/(self.kB_ev*T)) # from I. Takagi
                #self.kr_atom = 3.8e-26 * np.exp(-0.15/(self.kB_ev*T)) # from Mingzhong Zhao

                self.k_r = self.kr_atom * self.NA
                self.k_d = self.K_S**2 * self.k_r

                #self.D = 3.0e-6 * np.exp(-0.64/(self.kB_ev*T))
                #self.K_S = 1.1e-2 * np.exp(-0.17/(self.kB_ev*T))

            else:
                # 普通金属：缺 kd?
                if not kd_provided:
                    assert kr_provided, "缺 k_d 时必须提供 k_r0/E_kr"
                    self.k_r0, self.E_kr = k_r0, E_kr
                    self.k_d0 = K_S0**2 * k_r0
                    self.E_kd  = 2*E_KS + E_kr

                # 缺 kr?
                elif not kr_provided:
                    assert kd_provided, "缺 k_r 时必须提供 k_d0/E_kd"
                    self.k_d0, self.E_kd = k_d0, E_kd
                    self.k_r0 = k_d0/(K_S0**2)
                    self.E_kr = E_kd - 2*E_KS

                # 最后按 Arrhenius 算 kr，再 kd=KS²·kr
                self.k_d = self.arrhenius(self.k_d0, self.E_kd)
                self.k_r = self.arrhenius(self.k_r0, self.E_kr)

        self.compute_isotope_parameters()
        self.update_heteronuclear_parameters()
        
        # 计算初始的Delta_l和节点数N
        self.update_spatial_temporal_grid()

        self.print_initialization_summary()

    def calculate_HD_pressure(self):
        T = self.temperature
        K_eq_HD = 4.241 * ((1 - np.exp(-5986/T)) * (1 - np.exp(-4307/T))) \
                  / ((1 - np.exp(-5525/T)) ** 2) * np.exp(-78.7/T)
        return np.sqrt(K_eq_HD * self.H2_pressure * self.D2_pressure)

    def compute_isotope_parameters(self):
        beta = self.beta_factors['H_to_D'] if self.isotope_input == 'H' else self.beta_factors['D_to_H']
        self.D_iso = self.D * beta
        self.k_d_iso = self.k_d * beta
        self.k_r_iso = self.k_r * beta

    def arrhenius(self, P_0, E_P):
        """
        根据正指数形式的 Arrhenius 关系计算温度下的 P 值。
        E_P 单位为 kJ/mol，需乘1000转换为 J/mol。
        """
        return P_0 * np.exp(E_P * 1000 / (self.R * self.temperature))

    def update_heteronuclear_parameters(self):

        # 异质核气体参数 (ij)
        if self.isotope_input == 'H':
            self.k_d_ij = self.k_d * np.sqrt(2/3)
            self.k_r_ij = self.k_r * np.sqrt(2/3)
        else:
            self.k_d_ij = self.k_d * np.sqrt(4/3)
            self.k_r_ij = self.k_r * np.sqrt(4/3)

    def update_spatial_temporal_grid(self):
        """
        根据当前温度的扩散系数D计算空间步长Delta_l，
        并通过经验控制节点数N。
        """
        self.Delta_l = np.sqrt(self.D * self.Delta_t)
        self.N = int(self.L / self.Delta_l)

        if self.N > 500:
            self.N = 400  # 根据经验调整N到400
            self.Delta_l = self.L / self.N
            self.Delta_t = (self.Delta_l ** 2) / self.D
            print(f"节点数超过500，已调整为N=400，并重新计算Delta_t={self.Delta_t:.4e}s")
        elif self.N < 20:
            self.N = 20  # 根据经验调整N到400
            self.Delta_l = self.L / self.N
            self.Delta_t = (self.Delta_l ** 2) / self.D
            print(f"节点数少于20，已调整为N=20，并重新计算Delta_t={self.Delta_t:.4e}s")
        else:
            print(f"节点数N={self.N}，满足要求，无需调整Delta_t。")

    def print_initialization_summary(self):
        W = self.k_d * self.L * np.sqrt(self.D2_pressure) / (self.D * self.K_S)

        print(f"Simulation parameters initialized:")
        print(f"dimensionless permeation number is {W:.4f}")
        print(f"  H₂ Pressure: {self.H2_pressure} Pa")
        print(f"  D₂ Pressure: {self.D2_pressure} Pa")
        print(f"  HD Pressure: {self.HD_pressure:.3f} Pa")
        print(f"  Temperature: {self.temperature} K")
        print(f"  Gas phase equilibrium: {self.gas_phase_equilibrium}")
        print(f"  Material thickness L: {self.L} m")
        print(f"  Total simulation time: {self.total_time} s")
        print(f"  Initial time step Delta_t: {self.Delta_t:.4e} s")
        print(f"  Spatial step Delta_l: {self.Delta_l:.4e} m")
        print(f"  Number of nodes N: {self.N}\n")

        input_iso = self.isotope_input
        other_iso = 'D' if input_iso == 'H' else 'H'
        print(f"Calculated parameters at {self.temperature} K:")
        print(f"  {input_iso} isotope parameters:")
        print(f"    D = {self.D:.4e} m²/s")
        if self.K_S != None:
            print(f"    K_S = {self.K_S:.4e} mol/(m^3·pa^(0.5))")
            if self.kr_atom != None:
                print(f"    kr_atom = {self.kr_atom:4e} molec·m⁴/(s·at²)")
        print(f"    k_d = {self.k_d:.4e} mol/(m²·s·Pa)")
        print(f"    k_r = {self.k_r:.4e} m⁴/(s·mol)")
        print(f"  {other_iso} isotope parameters:")
        print(f"    D_iso = {self.D_iso:.4e} m²/s")
        print(f"    k_d_iso = {self.k_d_iso:.4e} mol/(m²·s·Pa)")
        print(f"    k_r_iso = {self.k_r_iso:.4e} m⁴/(s·mol)")

        print(f"\n  Heteronuclear molecule ij (HD) parameters:")
        print(f"    k_d_ij = {self.k_d_ij:.4e} mol/(m²·s·Pa)")
        print(f"    k_r_ij = {self.k_r_ij:.4e} m⁴/(s·mol)")


    def simulate(self):
        """
        使用有限差分法求解非稳态渗透问题，计算每个时间步各区块内同位素 i 和 j 的浓度分布。
        初始条件：t=0 时所有区块浓度均为 0。
        使用向量化操作计算内部区域的流 J，可以显著提高计算效率。
        """
        M = int(self.total_time / self.Delta_t)
        self.M = M
        N = self.N

        # 初始化浓度矩阵，单位为 mol/m³
        c_i = np.zeros((M + 1, N), dtype=np.float64)
        c_j = np.zeros((M + 1, N), dtype=np.float64)

        # 直接赋值扩散系数、表面参数（均已在 update_temperature_dependent_parameters 计算）
        D_i = self.D
        D_j = self.D_iso
        k_d_i2 = self.k_d
        k_d_j2 = self.k_d_iso
        k_r_i2 = self.k_r
        k_r_j2 = self.k_r_iso

        # 根据 isotype_input 设置对应气体分压
        if self.isotope_input == 'H':
            p_i2 = self.H2_pressure
            p_j2 = self.D2_pressure
        else:
            p_i2 = self.D2_pressure
            p_j2 = self.H2_pressure

        # 异质核气体参数
        k_d_ij = self.k_d_ij
        k_r_ij = self.k_r_ij
        p_ij = self.HD_pressure

        # 时间迭代
        for m in range(M):
            # 预分配当前时间步的净流量向量
            J_i = np.zeros(N, dtype=np.float64)
            J_j = np.zeros(N, dtype=np.float64)
            
            # 边界处理：第一块区域 (n = 0，对应公式中的 n=1)
            J_i[0] = (k_d_i2 * p_i2 + 0.5 * k_d_ij * p_ij -
                    k_r_i2 * (c_i[m, 0] ** 2) - 0.5 * k_r_ij * c_i[m, 0] * c_j[m, 0] -
                    (D_i * (c_i[m, 0] - c_i[m, 1])) / self.Delta_l)
            J_j[0] = (k_d_j2 * p_j2 + 0.5 * k_d_ij * p_ij -
                    k_r_j2 * (c_j[m, 0] ** 2) - 0.5 * k_r_ij * c_i[m, 0] * c_j[m, 0] -
                    (D_j * (c_j[m, 0] - c_j[m, 1])) / self.Delta_l)
            
            # 向量化计算内部区域（n=1 到 n=N-2）净流量
            if N > 2:  # 当至少存在内部节点时
                # 对于同位素 i
                c_left_i = c_i[m, 0:N-2]
                c_center_i = c_i[m, 1:N-1]
                c_right_i = c_i[m, 2:N]
                J_i[1:N-1] = D_i * (c_right_i + c_left_i - 2 * c_center_i) / self.Delta_l

                # 对于同位素 j
                c_left_j = c_j[m, 0:N-2]
                c_center_j = c_j[m, 1:N-1]
                c_right_j = c_j[m, 2:N]
                J_j[1:N-1] = D_j * (c_right_j + c_left_j - 2 * c_center_j) / self.Delta_l

            # 边界处理：最后一块区域 (n = N-1，对应公式中的 n=N)
            J_i[N-1] = (D_i * (c_i[m, N-2] - c_i[m, N-1])) / self.Delta_l - \
                    k_r_i2 * (c_i[m, N-1] ** 2) - 0.5 * k_r_ij * c_i[m, N-1] * c_j[m, N-1]
            J_j[N-1] = (D_j * (c_j[m, N-2] - c_j[m, N-1])) / self.Delta_l - \
                    k_r_j2 * (c_j[m, N-1] ** 2) - 0.5 * k_r_ij * c_i[m, N-1] * c_j[m, N-1]
            
            # 更新浓度（向量化更新）
            c_i[m + 1, :] = c_i[m, :] + (J_i / self.Delta_l) * self.Delta_t
            c_j[m + 1, :] = c_j[m, :] + (J_j / self.Delta_l) * self.Delta_t

            # 快速判据：检查第一个和第二个位置上的浓度差异(c_i[m + 1, 0] - c_i[m + 1, 1] <= 0)
            if (c_i[m+1, 0] ** 2 > 1e20) or (c_i[m + 1, 0] - c_i[m + 1, 1] <= 0) or (c_i[m+1, N-1] ** 2 > 1e20):  # 如果条件不满足
                # 调整步长：减少节点数 N，并重新计算空间步长 Delta_l
                # print(f"c_i[m + 1, 0]:{c_i[m + 1, 0]:.4e};\n c_i[m + 1, 1]:{c_i[m + 1, 1]:.4e}.")
                self.N -= 1  # 减少节点数
                self.Delta_l = self.L / self.N  # 重新计算空间步长
                print(f"节点数不满足条件，已调整为 N = {self.N}")

                # 从头开始模拟
                N = self.N
                c_i = np.zeros((M + 1, N), dtype=np.float64)
                c_j = np.zeros((M + 1, N), dtype=np.float64)
                m = 0  # 重新从时间步 m = 0 开始模拟
                continue  # 跳过当前时间步，重新开始计算

        # 根据输入同位素类型决定存储结果：将 c_i 和 c_j 分别归属为 c_H 和 c_D
        self.J_i2 = 0.5 * k_r_i2 * (c_i[:, N-1] ** 2)
        self.J_j2 = 0.5 * k_r_j2 * (c_j[:, N-1] ** 2)
        self.J_ij = 0.5 * k_r_ij * (c_i[:, N-1] * c_j[:, N-1])

        if self.isotope_input == 'H':
            self.c_H = c_i  # i 对应 H
            self.c_D = c_j  # j 对应 D
            self.J_H2 = self.J_i2
            self.J_D2 = self.J_j2
        else:
            self.c_D = c_i  # i 对应 D
            self.c_H = c_j  # j 对应 H
            self.J_D2 = self.J_i2
            self.J_H2 = self.J_j2

        self.J_HD = self.J_ij
        # 提取最后时刻（m=M）的浓度分布
        self.c_H_final = self.c_H[-1, :]
        self.c_D_final = self.c_D[-1, :]

        return self.c_H_final, self.c_D_final, self.J_H2, self.J_D2, self.J_HD

