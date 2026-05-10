import subprocess
from pathlib import Path

tex_content = r"""
\documentclass[12pt,a4paper]{article}

\usepackage{geometry}
\geometry{top=2.2cm,bottom=2.2cm,left=2cm,right=2cm}

\usepackage{xepersian}
\settextfont{Vazirmatn}
\setlatintextfont{Times New Roman}

\usepackage{amsmath,amssymb}
\usepackage{physics}
\usepackage{setspace}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{hyperref}

\linespread{1.5}

\titleformat{\section}
  {\large\bfseries\color{blue}}
  {\thesection.}{0.5em}{}

\begin{document}

\begin{center}
    {\LARGE \textbf{انواع پیوندها در جامدات و مفاهیم اولیه کشسانی}}\\[1em]
\end{center}

سلام و وقت بخیر.  
موضوع ارائه‌ی من درباره‌ی چند مبحث مهم در فیزیک حالت جامد است. در این ارائه ابتدا درباره‌ی
بلور گازهای بی‌اثر صحبت می‌کنم، سپس برهم‌کنش وان‌دروالس و نیروی لندن را توضیح می‌دهم.
بعد به‌ترتیب به پیوند یونی و انرژی الکترواستاتیکی، پیوند کووالانسی، پیوند فلزی و پیوند هیدروژنی
می‌پردازم. در پایان نیز به‌صورت خلاصه درباره‌ی مولفه‌های تنش، ثابت‌های سفتی و کشسانی صحبت خواهم کرد.

\section{مقدمه‌ای بر پیوند در جامدات}

در فیزیک حالت جامد، یکی از پرسش‌های اساسی این است که چرا اتم‌ها یا یون‌ها کنار هم قرار می‌گیرند
و یک جامد پایدار تشکیل می‌دهند. پاسخ این پرسش به نوع نیروها و پیوندهای بین ذرات بستگی دارد.

به‌طور کلی، جامدات می‌توانند توسط انواع مختلفی از پیوندها پایدار شوند. این پیوندها شامل پیوندهای
ضعیف مثل وان‌دروالس و هیدروژنی، و پیوندهای قوی‌تر مثل یونی، کووالانسی و فلزی هستند.

نوع پیوند، بسیاری از خواص فیزیکی ماده را تعیین می‌کند؛ برای مثال نقطه ذوب، سختی، رسانایی الکتریکی،
رسانایی گرمایی، شفافیت و رفتار کشسانی ماده به نوع پیوند و ساختار بلوری وابسته‌اند.

\section{بلور گازهای بی‌اثر}

گازهای بی‌اثر یا نجیب شامل عناصری مانند نئون، آرگون، کریپتون و زنون هستند. این اتم‌ها در حالت عادی
بسیار کم‌واکنش‌اند، زیرا لایه‌ی الکترونی بیرونی آن‌ها کامل است.

اما در دماهای پایین، این گازها می‌توانند جامد شوند و بلور تشکیل دهند. نیروی اصلی در این بلورها،
برهم‌کنش وان‌دروالس است.

پتانسیل بین دو اتم معمولاً با پتانسیل لنارد-جونز توصیف می‌شود:

\[
U(r)=4\varepsilon \left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right]
\]

در این رابطه، \(r\) فاصله بین دو اتم است.

\section{برهم‌کنش وان‌دروالس و نیروی لندن}

در گازهای بی‌اثر مهم‌ترین نوع برهم‌کنش وان‌دروالس، نیروی پراکندگی لندن است.

انرژی این نیرو تقریباً به‌صورت زیر به فاصله وابسته است:

\[
U(r)\propto -\frac{1}{r^6}
\]

در فاصله‌های کم نیز دافعه‌ی شدیدی به‌علت اصل طرد پائولی ایجاد می‌شود.

\section{پیوند یونی و انرژی الکترواستاتیکی}

نمونه‌ی کلاسیک پیوند یونی، بلور سدیم کلرید یا NaCl است.

انرژی کولنی بین دو یون برابر است با:

\[
U(r)=-\frac{e^2}{4\pi\varepsilon_0 r}
\]

در یک بلور یونی، انرژی کل معمولاً با ثابت مادلونگ بیان می‌شود:

\[
U=-\frac{N\alpha e^2}{4\pi\varepsilon_0 r_0}
\]

که در آن \(N\) تعداد جفت یون‌ها، \(\alpha\) ثابت مادلونگ و \(r_0\) فاصله نزدیک‌ترین همسایه‌ها است.

\section{پیوند کووالانسی}

پیوند کووالانسی زمانی تشکیل می‌شود که دو اتم یک یا چند جفت الکترون را به‌صورت مشترک استفاده کنند.

نمونه‌ی مهم آن الماس است. در الماس، هر اتم کربن با چهار اتم دیگر پیوند کووالانسی دارد و ساختاری
چهاروجهی ایجاد می‌کند. در این ساختار، کربن دارای هیبریداسیون \(sp^3\) است.

\section{پیوند فلزی}

در فلزات، الکترون‌های ظرفیت به اتم خاصی وابسته نیستند و می‌توانند در کل بلور حرکت کنند.
به این الکترون‌ها دریای الکترونی گفته می‌شود.

این ویژگی علت اصلی رسانایی الکتریکی بالا و چکش‌خواری فلزات است.

\section{پیوند هیدروژنی}

پیوند هیدروژنی زمانی ایجاد می‌شود که اتم هیدروژن به یک اتم بسیار الکترونگاتیو مانند اکسیژن،
نیتروژن یا فلوئور متصل باشد.

نمونه‌ی مهم آن آب و یخ است. در یخ، شبکه‌ای از پیوندهای هیدروژنی تشکیل می‌شود که باعث کاهش چگالی
یخ نسبت به آب مایع می‌شود.

\section{مولفه‌های تنش و ثابت‌های سفتی و کشسانی}

برای توصیف نیروهای داخلی در جامد از تنش استفاده می‌کنیم:

\[
\sigma=\frac{F}{A}
\]

در حالت کلی، تنش یک کمیت تانسوری است:

\[
\sigma_{ij}
\]

تانسور تنش به‌صورت زیر نوشته می‌شود:

\[
\begin{pmatrix}
\sigma_{xx} & \sigma_{xy} & \sigma_{xz}\\
\sigma_{yx} & \sigma_{yy} & \sigma_{yz}\\
\sigma_{zx} & \sigma_{zy} & \sigma_{zz}
\end{pmatrix}
\]

در تعادل مکانیکی معمولاً داریم:

\[
\sigma_{ij}=\sigma_{ji}
\]

رابطه‌ی کلی تنش و کرنش در محدوده‌ی خطی با قانون هوک تعمیم‌یافته بیان می‌شود:

\[
\sigma_{ij}=C_{ijkl}\epsilon_{kl}
\]

برای بلورهای مکعبی تنها سه ثابت مستقل کشسانی داریم:

\[
C_{11},\quad C_{12},\quad C_{44}
\]

\section{جمع‌بندی}

در این ارائه دیدیم که نوع پیوند در جامدات نقش بسیار مهمی در تعیین خواص فیزیکی و مکانیکی آن‌ها دارد.
بلورهای گازهای بی‌اثر با نیروهای وان‌دروالس پایدار می‌شوند، جامدات یونی با جاذبه کولنی، جامدات
کووالانسی با اشتراک الکترون، فلزات با دریای الکترونی و برخی مواد مولکولی با پیوند هیدروژنی.

همچنین دیدیم که خواص مکانیکی جامدات با مفاهیمی مانند تنش، کرنش و ثابت‌های کشسانی توصیف می‌شوند.

\vspace{1cm}
\begin{center}
\textbf{خیلی ممنون از توجه شما}
\end{center}

\end{document}
"""

tex_file = Path("solid_state_presentation.tex")
tex_file.write_text(tex_content, encoding="utf-8")

try:
    subprocess.run(
        ["xelatex", "-interaction=nonstopmode", str(tex_file)],
        check=True
    )
    print("PDF با موفقیت ساخته شد.")
except FileNotFoundError:
    print("خطا: xelatex روی سیستم نصب نیست.")
except subprocess.CalledProcessError:
    print("خطا در کامپایل فایل LaTeX.")
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Freeze Casting in Ceramics - Research Paper</title>
    
    <!-- Google Fonts for Academic Look -->
    <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Montserrat:wght@300;400;700&display=swap" rel="stylesheet">
    
    <!-- MathJax for Scientific Formulas -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        :root {
            --primary-color: #1a2a6c;
            --text-color: #2c3e50;
            --bg-color: #ffffff;
            --accent-color: #b21f1f;
        }

        body {
            font-family: 'Libre+Baskerville', serif;
            line-height: 1.8;
            color: var(--text-color);
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 1000px;
            margin: 50px auto;
            background: var(--bg-color);
            padding: 80px;
            box-shadow: 0 0 30px rgba(0,0,0,0.1);
        }

        header {
            text-align: center;
            border-bottom: 3px double #ccc;
            margin-bottom: 50px;
            padding-bottom: 20px;
        }

        h1 {
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
            font-size: 36px;
            color: var(--primary-color);
            margin-bottom: 10px;
        }

        .author-info {
            font-family: 'Montserrat', sans-serif;
            font-size: 14px;
            color: #777;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        h2 {
            font-family: 'Montserrat', sans-serif;
            font-size: 24px;
            color: var(--primary-color);
            border-left: 5px solid var(--accent-color);
            padding-left: 15px;
            margin-top: 50px;
        }

        h3 {
            font-family: 'Montserrat', sans-serif;
            font-size: 20px;
            color: #34495e;
            margin-top: 30px;
        }

        p {
            margin-bottom: 20px;
            text-align: justify;
        }

        .abstract-box {
            background: #f9f9f9;
            padding: 30px;
            border: 1px solid #eee;
            font-style: italic;
            margin-bottom: 40px;
        }

        .equation {
            background: #f0f4f8;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin: 30px 0;
            font-size: 1.2em;
        }

        .figure-container {
            margin: 40px 0;
            text-align: center;
        }

        .figure-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        .caption {
            font-size: 14px;
            color: #555;
            margin-top: 15px;
            font-style: italic;
            padding: 0 10%;
        }

        .highlight-box {
            border-left: 5px solid #2ecc71;
            background: #f2fdf5;
            padding: 20px;
            margin: 20px 0;
        }

        footer {
            margin-top: 80px;
            font-size: 12px;
            color: #999;
            text-align: center;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }

        ul {
            padding-left: 20px;
        }

        li {
            margin-bottom: 10px;
        }

        @media print {
            body { background: white; }
            .container { box-shadow: none; margin: 0; padding: 40px; width: 100%; }
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>Advanced Freeze Casting in Ceramic Engineering: <br>Mechanisms, Models, and Industrial Applications</h1>
        <div class="author-info">Prepared by: ایلیا | Materials & Data Science Engineer</div>
    </header>

    <div class="abstract-box">
        <strong>Abstract:</strong> Freeze casting (ice templating) has revolutionized the production of porous ceramics by allowing for unprecedented control over pore architecture. This paper explores the complex thermodynamics governing particle rejection during directional solidification and the resulting anisotropic mechanical properties. We analyze the growth kinetics of ice crystals and their role as sacrificial templates. Finally, industrial applications in biotechnology, energy, and aerospace are discussed, emphasizing the role of AI-driven structural simulations in future materials design.
    </div>

    <h2>1. Introduction: The Evolution of Porous Ceramics</h2>
    <p>
        The fabrication of porous ceramics with tailored architectures is a cornerstone of modern materials science. Traditional methods, while effective for bulk porosity, often lack the precision required for high-end applications like bone tissue scaffolds or fuel cell electrodes. 
    </p>
    
    <div class="figure-container">
        <img src="comparison infographic of porous ceramic fabrication methods freeze casting, foam replication, sacrificial template, showing pore structure differences.png" alt="Comparison of Methods">
        <div class="caption"><strong>Figure 1:</strong> Scientific comparison of fabrication methods. Note the superior anisotropy and connectivity of <strong>Freeze Casting</strong> compared to traditional foam replication and random sacrificial templates.</div>
    </div>

    <p>
        As shown in Figure 1, <strong>Freeze Casting</strong> (also known as ice-templating) stands out due to its ability to produce highly aligned, lamellar channels. Unlike foam replication, which creates a reticulated network, freeze casting utilizes the physics of ice growth to push ceramic particles into densified walls, creating a truly anisotropic material.
    </p>

    <h2>2. Thermodynamic Fundamentals of Solidification</h2>
    <p>
        The core of freeze casting is the interaction between the advancing solidification front (the ice-liquid interface) and the suspended ceramic particles. This process is governed by the change in Gibbs Free Energy:
    </p>

    <div class="equation">
        \[ \Delta G_{solidification} = \Delta H - T \Delta S \]
    </div>

    <p>
        For a particle to be <strong>rejected</strong> (pushed) by the ice front rather than <strong>engulfed</strong>, a thermodynamic repulsive force must exist between the ice and the particle. This is defined by the critical velocity \( V_c \). If the freezing front velocity \( V \) is less than \( V_c \), the particle is pushed:
    </p>

    <div class="equation">
        \[ V_c = \frac{d \Delta \gamma}{3 \eta R} \left( \frac{a_0}{d} \right)^z \]
    </div>

    <p>
        Where \( \Delta \gamma \) is the surface energy difference, \( \eta \) is the viscosity of the slurry, \( R \) is the particle radius, and \( d \) is the thickness of the liquid film between the particle and the ice.
    </p>

    <div class="figure-container">
        <img src="scientific diagram of particle rejection during ice front propagation in freeze casting, ceramic particles pushed between growing ice crystals, materials science illustration.png" alt="Particle Rejection Mechanism">
        <div class="caption"><strong>Figure 2:</strong> The physics of particle rejection. As the freezing direction proceeds downwards, ceramic particles are concentrated into "Particle-Rich Ceramic Wall" regions, forming the lamellar backbone of the final material.</div>
    </div>

    <h2>3. Ice Growth Models and Microstructure Evolution</h2>
    <p>
        The morphology of the resulting pores is a direct mirror of the ice crystal growth. In aqueous systems, ice grows preferentially along the a-axis, creating hexagonal or lamellar shapes depending on the supercooling degree. 
    </p>

    <div class="highlight-box">
        <strong>The Spacing Law:</strong> The lamellar spacing \( \lambda \) is inversely proportional to the freezing velocity \( V \). This is one of the most critical equations for an engineer to control the microstructure:
        <div class="equation">
            \[ \lambda = K V^{-n} \]
        </div>
        Where \( n \) is typically 0.5. To get smaller pores, ایلیا، you must increase the cooling rate significantly!
    </div>

    <div class="figure-container">
        <img src="SEM microstructure of freeze cast porous ceramic showing aligned lamellar pores, high resolution scanning electron microscopy, materials science microstructure.png" alt="SEM Microstructure">
        <div class="caption"><strong>Figure 3:</strong> High-resolution Scanning Electron Microscopy (SEM) of a freeze-cast ceramic. The perfectly aligned lamellar pores and sintered ceramic bridges are clearly visible, illustrating the result of the spacing law.</div>
    </div>

    <h2>4. Engineering Properties: Anisotropy and Strength</h2>
    <p>
        Because the pores are aligned, the mechanical properties are highly <strong>anisotropic</strong>. The compressive strength \( \sigma \) is significantly higher when loaded parallel to the freezing direction compared to perpendicular loading. 
    </p>

    <div class="figure-container">
        <img src="3D illustration of anisotropic porous ceramic structure created by freeze casting, aligned channels and lamellar walls, engineering visualization.png" alt="3D Porous Structure">
        <div class="caption"><strong>Figure 4:</strong> 3D visualization of the final ceramic structure. This architecture allows for high fluid permeability along the Z-axis while maintaining structural integrity.</div>
    </div>

    <h2>5. Industrial and Functional Applications</h2>
    
    <h3>5.1 Biomedical: Bone Scaffolds</h3>
    <p>
        The lamellar structure produced by freeze casting is strikingly similar to the architecture of <strong>trabecular bone</strong>. Using Hydroxyapatite (HAp) or Bio-glass, engineers can create scaffolds that encourage bone ingrowth and nutrient transport through the aligned channels.
    </p>

    <h3>5.2 Energy: SOFC and Batteries</h3>
    <p>
        In Solid Oxide Fuel Cells (SOFC), the efficiency is limited by gas diffusion. Aligned pores reduce the tortuosity factor \( \tau \), allowing oxygen and fuel to reach the triple phase boundaries (TPB) much faster:
    </p>
    <div class="equation">
        \[ D_{eff} = \frac{\epsilon}{\tau} D \]
    </div>
    <p>By minimizing \( \tau \) through freeze casting, the power density of the cell increases by up to 40%.</p>

    <h3>5.3 Filtration and Catalysis</h3>
    <p>
        Industrial filters for molten metals or wastewater benefit from the low pressure drop across freeze-cast ceramics. The high surface area-to-volume ratio also makes them ideal supports for catalytic coatings in chemical reactors.
    </p>

    <h2>6. Conclusion and Future Perspectives</h2>
    <p>
        ایلیا، as a materials and data science engineer, the next frontier in freeze casting is the <strong>AI-driven optimization</strong> of the slurry rheology. By simulating the particle interaction using Python libraries like <i>FIPY</i> or <i>PySPH</i>, we can predict the exact \( V_c \) and design ceramics that were previously impossible to manufacture. 
    </p>
    <p>
        Recent advances in 2024-2025 have already seen the integration of <strong>Additive Manufacturing (3D Printing)</strong> with freeze casting, allowing for hierarchical macro-porosity and directional micro-porosity in a single component.
    </p>

    <footer>
        &copy; 2026 | Professional Research Report for Iliya | Materials Science & Machine Learning Division
    </footer>
</div>

</body>
</html>
