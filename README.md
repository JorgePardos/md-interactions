# md_interactions

Análisis reproducible de interacciones y **geometría de reacción** en trayectorias de
dinámica molecular (`prmtop` de AMBER + `.nc`/`.dcd`), pensado para sistemas
enzima–sustrato: distancias catalíticas, ángulos de ataque, puentes de hidrógeno,
RMSD/RMSF, mapas 2D de energía libre aparente, Rg, RDF y clustering conformacional.

Apuntas la herramienta a un par (topología, trayectoria) + un YAML con las
selecciones que te interesan y obtienes una carpeta `results/` con figuras
listas para publicación (PNG + PDF/SVG), CSVs con todos los datos y un informe
único en Markdown/HTML.

```
results/
├── plots/        dist_d_nuc_timeseries.png/.pdf, fes_dnuc_dacid.png, rmsf_per_residue.png ...
├── data/         distances.csv, angles.csv, rmsd.csv, hbonds_tracked.csv, observables.csv ...
├── summary/      summary.csv, summary.md, clusters.csv
├── structures/   cluster1_frame1834.pdb  (representantes de cada clúster)
├── report.md
└── report.html
```

---

## Instalación

```bash
pip install -e .
```

Requiere Python ≥ 3.10 y depende de MDAnalysis, numpy, pandas, matplotlib, scipy y
PyYAML. Extras opcionales:

```bash
pip install -e ".[netcdf]"   # netCDF4 (opcional)
pip install -e ".[dev]"      # pytest
```

Los `.nc` de AMBER se leen con `scipy.io.netcdf` (viene con scipy), así que **no**
necesitas `netCDF4` para analizar; solo acelera la *escritura* de netCDF, que aquí
no se usa.

> El clustering usa `scipy` (jerárquico y k-means), así que **no** hace falta
> scikit-learn.

---

## Interfaz gráfica: `md-analyzer gui`

Un único comando, una ventana, sin YAML:

```bash
md-analyzer gui --top system.prmtop --traj prod.nc
```

Se abre el navegador con la estructura en 3D. **Pinchas los átomos** que quieres
medir (2 → distancia, 3 → ángulo, 4 → diedro), le das a analizar y las figuras
aparecen en la misma ventana. Incluye un botón de detección automática que
propone las interacciones que ya existen en la trayectoria, y otro que busca
cambios químicos (enlaces rotos/formados, protones migrados).

En un cluster, con reenvío de puerto:

```bash
ssh -L 8765:localhost:8765 usuario@cluster
md-analyzer gui --top system.prmtop --traj prod.nc --no-browser
```

El visor 3D (3Dmol.js) va **empaquetado**, así que funciona sin internet. El
servidor es de la biblioteca estándar: no añade dependencias.

## Detección automática: `md-analyzer explore`

Invierte el problema: en vez de decirle tú qué medir, te dice qué hay.

```bash
md-analyzer explore contacts --top x.prmtop --traj y.nc --around "resname LIG"
md-analyzer explore changes  --top x.prmtop --traj y.nc
```

`contacts` lista los contactos polares, puentes salinos y de hidrógeno de una
región con su ocupación, ordenados. `changes` detecta enlaces covalentes que se
rompen o se forman y protones que cambian de átomo — es decir, te avisa de que
la trayectoria es reactiva y de que **la topología describe solo la estructura
inicial**, con lo que una selección por nombre de átomo puede no estar midiendo
lo que su nombre sugiere. Con `--export config.yaml` lo detectado se convierte
en una configuración lista para `run`.

## Uso desde la línea de comandos

```bash
md-analyzer wizard --top system.prmtop --traj prod.nc   # asistente interactivo
md-analyzer check  -c config.yaml    # valida el YAML y TODAS las selecciones
md-analyzer run    -c config.yaml    # ejecuta los análisis activados
```

### Asistente: `md-analyzer wizard`

La forma recomendada de crear la configuración sin escribir YAML a mano. Lee la
topología, te deja explorarla y valida cada selección **en el momento**:

```
  Átomo/grupo A: ?ARG
  ARG: 26 residuo(s) -> resid [26, 29, 33, ..., 414]
    átomos del primero: N, H, CA, HA, CB, ..., NH1, HH11, HH12, NH2, HH21, HH22
  Átomo/grupo A: 20@OD1
  [ok] ASP20:OD1 (1 átomo)
  Átomo/grupo B: ARG20@HH12
  [!] El residuo 20 es ASP, no ARG. Usa '?ARG' para listar los ARG de la topología.
  Átomo/grupo B: 414@HH12
  [ok] ARG414:HH12 (1 átomo)
```

Sintaxis de selección admitida:

| Escribes | Significa |
| --- | --- |
| `20@OD1` | resid 20, átomo OD1 |
| `ASP20@OD1` | igual, comprobando que el residuo 20 es un ASP |
| `TRH@O2P` | por nombre de residuo (ligando, cofactor...) |
| `20@OD1,OD2` | varios átomos del mismo residuo |
| `resid 20 and name OD1` | selección MDAnalysis literal, por si la necesitas |

Y consultas para explorar la topología sin salir del asistente: `?20` (muestra el
residuo y sus átomos), `?ARG` (lista todos los ARG), `?ligandos` (residuos no
proteicos ni disolvente), `?` (ayuda).

El menú principal lista **todos** los análisis disponibles con una descripción de
una línea, así que no hace falta leer la plantilla entera para saber qué existe.
Al terminar escribe un `config.yaml` normal —editable y versionable— con los
alias generados a partir de lo que realmente casó (`ASP20_OD1`, no `20_OD1`).

Plantillas en crudo, si prefieres partir de un fichero:

```bash
md-analyzer init  -o config.yaml     # plantilla comentada (trayectoria)
md-analyzer init --data -o config_data.yaml   # plantilla para tablas .dat
```


`check` es el paso que ahorra tiempo: carga el sistema, resuelve cada selección e
imprime cuántos átomos encuentra cada una, sin recorrer la trayectoria.

```
Selections:
  [ok]   distance:d_nuc               SER160_OG              ->      1 atoms
  [ok]   distance:d_nuc               LIG_C                  ->      1 atoms
  [FAIL] distance:d_acid              HIS237_NE2             -> "resid 237 and name NE2"
         Selection 'distance:d_acid:HIS237_NE2' -> "resid 237 and name NE2" matched 0 atoms.
```

Cualquier opción del YAML se puede sobrescribir desde la CLI, útil para lanzar la
misma configuración sobre varias réplicas:

```bash
md-analyzer run -c config.yaml --traj rep1/prod.nc rep2/prod.nc --out results_rep12 --stride 10
```

Opciones: `--top`, `--traj` (varios archivos), `--out`, `--stride`, `--start`,
`--stop`, `--formats png pdf svg`, `--strict` (aborta al primer fallo en vez de
continuar con el resto de análisis), `-q`.

---

## Uso como librería (notebooks)

```python
import md_interactions as mdi

config = mdi.load_config("config.yaml")
out = mdi.run_analyses(config)

out.summary                                  # DataFrame: media ± sd, min, max
out.observables                              # todas las series temporales juntas
out.result("distances").tables["distances"]  # DataFrame frame/time/d_nuc/d_acid
out.result("hbonds").plots                   # rutas de las figuras
```

Y módulo a módulo, sin escribir nada en disco:

```python
from md_interactions.analyses import distances, rmsd_rmsf

system = mdi.load_system(config)             # topología + trayectoria + stride
df = distances.compute_distances(system)     # DataFrame listo para pandas/seaborn
per_atom, per_residue = rmsd_rmsf.compute_rmsf(system)
```

Para probar sin datos propios:

```bash
python examples/toy_demo.py     # sistema de juguete + todos los análisis
```

---

## El archivo de configuración

Plantilla completa y comentada en [`examples/config_example.yaml`](examples/config_example.yaml)
(o `md-analyzer init`). Estructura resumida:

```yaml
system:
  topology: system.prmtop
  trajectory: [prod_01.nc, prod_02.nc]     # se concatenan (réplicas o tramos)
  time:   {dt: 0.01, unit: ns}             # dt entre frames GUARDADOS; null -> del .nc
  frames: {start: 0, stop: null, stride: 1}
  align:  {enabled: false, selection: "protein and name CA"}

output:
  directory: results
  formats: [png, pdf]                      # pdf/svg = vectorial para publicación
  dpi: 300

selections:                                # alias reutilizables (sintaxis MDAnalysis)
  SER160_OG:  "resid 160 and name OG"
  LIG_C:      "resname LIG and name C1"
  ACTIVE_SITE: "byres (protein and around 6 resname LIG)"

analyses:
  distances:
    pairs:
      - {name: d_nuc, atoms: [SER160_OG, LIG_C], threshold: 3.5}
      - {name: d_wat, atoms: [LIG_C, WATERS], mode: min}
  angles:
    definitions:
      - {name: a_attack, atoms: [SER160_OG, LIG_C, LIG_O]}
  ...
report:
  formats: [markdown, html]
```

Notas de diseño que conviene conocer:

- **Alias o selección literal**: donde se espera una selección puedes poner el nombre
  definido en `selections:` o directamente la cadena MDAnalysis
  (`"resid 160 and name OG"`). Los alias se usan también en las etiquetas de las figuras.
- **`mode`** en distancias/ángulos: `atom` (por defecto, la selección debe dar
  exactamente 1 átomo), `com` (centro de masas del grupo) o `min` (distancia mínima
  entre los dos grupos: el agua/oxígeno más cercano de todos).
- **`threshold`** en una distancia añade al resumen el % de frames por debajo del
  umbral (población de conformaciones near-attack) y lo dibuja en la serie temporal.
- **Claves desconocidas dan error** en vez de ignorarse en silencio: una errata en
  el YAML se detecta antes de leer un solo frame.
- **Nombres únicos**: cada observable (`d_nuc`, `a_attack`, …) es una columna del CSV
  global y el identificador que usan los mapas 2D como eje.

---

## Réplicas: no las mezcles a ciegas

Si listas varias trayectorias en `trajectory:` se concatenan como una sola serie.
Cuando son **réplicas independientes**, decláralas como tales:

```yaml
system:
  topology: system.prmtop
  replicas:
    rep1: [rep1/prod.nc]
    rep2: [rep2/prod.nc]
    rep3: [rep3/prod.nc]
```

Con eso, cada frame conserva de qué réplica viene (columna `replica` en todos los
CSV), las distribuciones llevan una curva discontinua por réplica sobre el
histograma agregado, y se genera `summary/replica_overlap.csv` + un mapa de calor
con el **coeficiente de solapamiento** de cada par (1 = distribuciones idénticas,
0 = disjuntas). Si algún par baja de 0.70 el informe lo avisa explícitamente.

> No se dan p-valores a propósito: los frames consecutivos están correlacionados,
> así que un test tipo Kolmogórov–Smirnov asume un tamaño muestral efectivo que
> una trayectoria no tiene y acaba declarando "significativo" casi todo.

Esto es justo lo que destapa el caso típico: una población que aparece en el
histograma conjunto pero que en realidad solo visita una de las réplicas.

## Datos ya calculados (`data:`) — sin topología ni trayectoria

Para analizar salidas de `cpptraj` (o de cualquier código que escriba una fila por
frame), incluidas simulaciones que no son MD de AMBER:

```yaml
data:
  replicas:
    rep1: [dist_rep1.dat]
    rep2: [dist_rep2.dat]
    rep3: [dist_rep3.dat]
  unit: "Å"
  time: {dt: 0.1, unit: ps}     # null -> el eje x son frames
  labels: {"D20:OD1-R32:HH12": d_saltbridge}   # renombrado opcional
  thresholds: {d_saltbridge: 2.5}              # % de frames por debajo
  normalization: density
```

Formato esperado (el de `cpptraj distance`):

```
#Frame     D20:OD1-R32:HH12   D20:OD2-R32:HH22
       1             2.5962             2.3542
```

Se valida que **todas las réplicas tengan las mismas columnas en el mismo orden**
(si no, aborta en vez de mezclar observables en silencio). Produce las mismas
distribuciones, tabla resumen, mapas 2D, comparación entre réplicas e informe que
el modo trayectoria. Desde notebook:

```python
import md_interactions as mdi
dataset = mdi.load_dataset(mdi.load_config("config_data.yaml"))
dataset.data          # DataFrame frame/time/replica/columnas
```

## Análisis disponibles

| Módulo | Qué produce |
| --- | --- |
| `distances` | Serie temporal + histograma/KDE por distancia, **rejilla multipanel** con todas las distribuciones; media ± sd, min/max, % bajo umbral |
| `distributions` | Lo mismo sobre tablas ya calculadas (`data:`), sin trayectoria |
| `replicas` | Solapamiento de distribuciones entre réplicas, tablas de convergencia y aviso de divergencia |
| `angles_dihedrals` | Ángulos [0,180]° y diedros (−180,180]°; estadística **circular** para diedros |
| `rmsd_rmsf` | RMSD global y local (ajuste sobre una selección, medida sobre otra) + RMSF por residuo |
| `hbonds` | Distancia D–A, ángulo D–H···A y ocupación (%) de puentes concretos + detección automática en una región |
| `free_energy_map` | Histograma 2D o KDE de dos observables; opción `−kT ln P` con barra de color en kcal/mol, kJ/mol o kT |
| `radius_of_gyration` | Rg de una o varias selecciones |
| `rdf` | g(r) entre dos selecciones + número de coordinación acumulado n(r) |
| `clustering` | Clustering jerárquico o k-means sobre RMSD del sitio activo; poblaciones, proyección PCA y **PDB del frame representativo** de cada clúster |
| `report` | Tabla resumen (CSV/Markdown) e informe único Markdown + HTML con todas las figuras |

Cada análisis se activa/desactiva con su `enabled:` (y se desactiva solo si no
declaras entradas). Un fallo en un módulo —típicamente una selección vacía— se
reporta y **no impide** que el resto se ejecute; con `--strict` se aborta.

### Detalles que afectan a la interpretación

- **RMSD local**: `superposition:` define sobre qué se hace el ajuste y `selection:`
  qué se mide. Lo habitual para un ligando o el sitio activo es ajustar sobre
  `backbone` y medir sobre `resname LIG`.
- **RMSF**: cada frame se superpone sobre la estructura promedio (ajuste iterativo de
  2 pasadas) antes de calcular las fluctuaciones; el valor por residuo es la media
  cuadrática ponderada por masa de sus átomos. Solo se cargan en memoria las
  coordenadas de la selección implicada.
- **Puentes de hidrógeno**: si el aceptor abarca varios átomos (p. ej. los dos
  oxígenos de un carboxilato) se toma el más cercano en cada frame, y de los
  hidrógenos del donor el que da el ángulo más lineal. La ocupación es el % de frames
  que cumplen ambos criterios (por defecto d ≤ 3.5 Å y ∠ ≥ 150°).
- **Mapas 2D**: `−kT ln P` referido al bin más poblado. En MD clásica sin sesgo esto
  es un mapa de poblaciones, **no** una superficie de energía libre convergida; los
  bins no muestreados se dejan en blanco. Si vienes de metadinámica/umbrella sampling,
  reponderar antes. El número de bins se recorta automáticamente (regla √(n/2)) si hay
  pocos frames para la rejilla pedida —se avisa por pantalla y en el informe—, y
  `smooth: <sigma en bins>` aplica suavizado gaussiano al histograma.
- **Clustering**: la matriz de RMSD por pares se calcula tras superponer todos los
  frames sobre la estructura promedio (RMSD = |xᵢ − xⱼ|/√N). Por encima de
  `max_frames` (2000 por defecto) se submuestrea, porque el coste es O(N²). El
  representante de cada clúster es el **medoide**, que se escribe como PDB del
  sistema completo (aguas y caja incluidas) — directamente usable como punto de
  partida QM/MM. Con `write_selection:` se recorta a una región (p. ej.
  `"byres (around 6 resname LIG)"`) si solo quieres inspeccionarlo en un visor.
- **PBC**: en distancias y ángulos se aplica convención de imagen mínima si la
  trayectoria trae caja válida (`pbc: false` para desactivarlo).
- **Histogramas**: por defecto en **densidad de probabilidad** (Å⁻¹). Es la única
  normalización comparable entre paneles cuando cada uno tiene su propio rango:
  con 30 bins fijos, un panel de 0.8 Å de rango usa bins de 0.026 Å y otro de 12 Å
  los usa de 0.4 Å, así que un "20 % por bin" significa cosas distintas en cada
  uno. La densidad integra a 1 y por eso el pico **puede superar 1 Å⁻¹** (lo hace
  siempre que σ < 0.40 Å). Alternativas: `normalization: percent | counts`.
- **Rejilla multipanel** (`facet: true`, por defecto): una sola figura con un panel
  por observable, con el número de columnas elegido automáticamente para que quepa
  en una página (≤ 9 in de alto), en vez de una figura por distancia.

---

## Estructura del paquete

```
src/md_interactions/
├── config.py        dataclasses + carga/validación del YAML
├── system.py        Universe, stride, eje temporal, selecciones, réplicas
├── tabular.py       lectura de tablas ya calculadas (cpptraj .dat, CSV)
├── plotting.py      estilo común, paleta (Okabe–Ito), guardado multi-formato
├── io_utils.py      árbol results/, CSV, tablas Markdown
├── results.py       AnalysisResult (tablas, series, resumen, figuras)
├── runner.py        orquestador
├── report.py        resumen + informe Markdown/HTML
├── cli.py           wizard / run / check / init
├── wizard.py        asistente interactivo, resolución de selecciones
├── testing.py       sistema de juguete sintético
└── analyses/        distances, angles_dihedrals, rmsd_rmsf, hbonds,
                     free_energy_map, radius_of_gyration, rdf, clustering,
                     distributions (modo data:), replicas (convergencia)
```

Todos los módulos de `analyses/` siguen el mismo contrato:

```python
run(system, paths, config=None, verbose=True) -> AnalysisResult   # escribe CSV + figuras
compute_*(system, config=None) -> pandas.DataFrame                # solo cálculo
```

---

## Tests

```bash
pytest
```

60 tests sobre un sistema sintético de 42 átomos generado al vuelo
(`md_interactions.testing`): no hacen falta trayectorias de prueba en el repo. Cubren
validación de configuración, manejo de frames/stride/alineamiento, comprobación
numérica de distancias/ángulos/RMSD/Rg contra cálculos directos con numpy, ocupación
de puentes de hidrógeno, escalado del mapa de energía libre con la temperatura,
clustering (poblaciones + medoides) y el flujo completo CLI → informe.

---

## Convenciones de unidades

| Magnitud | Unidad |
| --- | --- |
| Distancias, RMSD, RMSF, Rg, r de g(r) | Å |
| Ángulos y diedros | ° |
| Tiempo | ns por defecto (`ps` o `frame` configurables) |
| Energía libre | kcal/mol (o kJ/mol, kT) |
