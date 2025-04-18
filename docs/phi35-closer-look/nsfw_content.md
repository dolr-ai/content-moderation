# Error Analysis for NSFW_CONTENT

## False Negatives (actual=nsfw_content, predicted=clean): 208

```
Text: I love feeling your heart race when we're together.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I love the way you push deeper, hitting all the right spots.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Can you tell me your fantasy?
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Your words always make me feel cherished.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I love sighing softly with satisfaction, feeling completely content.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

## False Positives (predicted=nsfw_content): 499

### Breakdown by actual category:

#### Actual: spam_or_scams (Count: 319)
```
Text: hi

at last viagra , xenical ( and many other perscription drugs ) online !
no waiting rooms , drug stores ,
or embarassing conversations .
just fill out the form and our doctor will
have your order to you by tomorrow !
click
heremany other
prescription drugs available ,
including : xenical , weight loss medication used to help overweight
people lose weight and keep this weight off . valtrex , treatement
for herpes .
propecia , the first pill that effectively treats male pattern hair
loss .
and many more . . . . . .
click here
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: xe moment of great saving on pc programs .

we are your great solution for quality pc program discs
what would you do if you can choose to save hundreds on pc program discs for office operation , programming , server maintenance , pc diagnostics , office administration , finance and graphic design
font - family : arial ' >
to save hundreds on program installation and system upgrade , you should choose our site and get pc program discs for office operation , programming , server maintenance , pc diagnostics , office administration , finance and graphic design
font - family : arial ' >
our discount site have low priced pc program discs for your selection
vick and what theyll do to stop him . from reid on down , they said theyll take a day to
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: this service is provided by licensed internet pharmcy volition

enjoy lowpriced m = eds as our customer
rx meds for allergy , wt . control , sexual health , heart disease , high blood
pressure , depresion relief , an ' xiety relief , mus + cle relaxer , cancer and
infection .
meds affordable cause it is internet http : / / iwo . net . medwaftgood . com
get on the road towards internet pharamcy
take a chance on megonna do my very best and it ain ' t no lie
yeah , tell it , tell it , tell it like it t - i - is
errorbulletin aodc discusions hzol fnbcnet dnostdhdrs
you gotta move ityou gotta move it , move it , move it
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: just cents on the dollar : norton system works , adobe , dreamweaver , corel , windows xp , office xp - v 315 ln 39

s 627 lp 85
at fraction of retail prices : norton system works , adobe , dreamweaver , corel , windows xp , office xp - so 87 dq 50
ya got to see this - http : / / valet . lmbgfaeh . com / ? 5 qda 7 sb 2 fac . n 5 bmensuration
no more - http : / / dartmouth . ebidkdeb . com / pazthimbu ? 2 h 47 ajy _ 6 dfxqyygallows
k 992 lo 33
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: increase the volume of your ejaculation .

heya !
has your cum ever dribbled and you wish it had shot out ?
have you ever wanted to impress your girl with a huge cumshot ?
spur - m is the only site to offer an all natural male enhancement
formula that is proven to increase your sperm volume by up to 500 % .
our highly potent , volume enhancing formula will give our results
in days and comes with an impressive 100 % guarantee .
imagine the difference ( look and feel ) between dribbling your cum
compared to shooting out burst after burst . try spur - m now ! and
with our money back guarantee you have absolutely nothing to lose !
look here : http : / / drizzlier . com / cum /
no thanks : http : / / drizzlier . com / rr . php
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Actual: clean (Count: 44)
```
Text: Aphrodisiacs 

Recipe for an aphrodisiac:

Glaze a chopped onion in butter
Add three handfuls of nettles
Steam for three minutes
Add a pint of milk and nearly boil
Liquidise

Cooking destroys the sting but even so, drink quickly to avoid a stiff neck  20:33, 21 Mar 2005 (UTC)
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I know where the  &lt;#&gt;  is, I'll be there around 5
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: "Im at arestaurant eating squid! i will be out about 10:30 wanna dosomething or is that to late?"
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: holiday party - save the date

please click on the link below
save the date
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: ^ Have you not seen the show? Blobby was a name they used to cover 'Dildo' during pre-watershed slots, Dilso was the real name.
Confidence Score: 0.600
--------------------------------------------------------------------------------
```


#### Actual: offensive_language (Count: 115)
```
Text: @SeauxCocoa you getting new pussy and not sharing??
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: When the ghost picks u up after u dive in the pussy http://t.co/vBD4zHJROF
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I wan do a mixtape... "pussy an cinnamon rolls"
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: @DaveSwerve lmaoooo middle school knocking down all the pussy with the earrings on my shirt and no French toast
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Hugh Heffner gonna be on Life Support, STILL gettin pussy
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Actual: hate_or_discrimination (Count: 21)
```
Text: Eagles fuck around &amp; lose it'll be kill the cracker at the Sophi crib smfh
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: faggots http://t.co/DMKbGhai1z
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Good Mernin faggots http://t.co/3ZPTPqFkTl
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Cute ass niggah. http://t.co/h0SQoPTYfX
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: they shot another monkey lmaooooooooooo
Confidence Score: 0.900
--------------------------------------------------------------------------------
```
