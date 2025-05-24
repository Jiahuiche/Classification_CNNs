# Laboratori XNDL-10 notat

L'objectiu és crear un model des de zero (sense utilitzar cap pes pre-entrenat) capaç de classificar imatges de 28x28 píxels en una de les 12 categories següents: poma, rellotge, brúixola, nabiu, cervell, pilota de bàsquet, pilota de baseball, pilota de futbol, careta somrient, patata, octagon, cercle.

data/
├── train/          ← imatges del dataset original (Google QuickDraw)
├── validation/     ← dibuixos fets per vosaltres
└── test/           ← ocult, però idèntic a validation en procedència

Totes les imatges són de 28x28 i blanc i negre.

## Instruccions

- Entreneu el vostre model des de zero. No es permet cap pes preentrenat.
- L’entrenament ha de durar com a màxim 10 minuts en CPU. El vostre codi serà executat per mi en una màquina prou potent.
- El fitxer final ha de ser un únic .py, amb codi lliure però llegible. Siusplau comenteu el vostre codi.
- Es proporciona un esquelet amb els elements bàsics. Es pot modificar tot, sempre i quant compleixi les restriccions.
- Es fa en parelles (s’assumeixen els grups de la pràctica anterior). Si voleu canviar de grup, ho podeu dir als laboratoris la setmana vinent.
- Data límit: Dissabte 31 a les 23:59

## Avaluació

He creat una baseline que obté els següents números:

- Train (subset) Accuracy: 70.10%
- Validation (final) Accuracy: 73.91%
- Test Accuracy: 73.03%

Per aprovar (nota ≥ 5) necessiteu treure més d'un 70% d'accuracy (definit com micro F1) en test (mateixa distribució que el valid proporcionat). El 10 el marcarà qui aconsegueixi treure la millor puntuació. La resta de notes seran proporcionals al màxim i al 5 definit com un 70% d'accuracy.

Ànim i bona feina!

