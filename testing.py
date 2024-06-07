import math
import pickle
from staliro.core import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace
from staliro.staliro import staliro
from staliro.options import Options
import numpy as np
from numpy.typing import NDArray

from lsemibo.coreAlgorithm import LSemiBOOptimizer
from lsemibo.gprInterface import InternalGPR
from lsemibo.classifierInterface import InternalClassifier

from staliro.staliro import staliro

NLFDataT = NDArray[np.float_]
NLFResultT = ModelResult[NLFDataT, None]


""" class NLFModel(Model[NLFResultT, None]):
    def simulate(
        self, static: ModelInputs, intrvl: Interval
    ) -> NLFResultT:
        print(static)
        timestamps_array = np.array(1.0).flatten()
        X = static.static[0]
        Y = static.static[1]
        d1 = X**3
        d2 = math.sin(X/2) + math.sin(Y/2) + 2
        d3 = math.sin((X-3)/2) + math.sin((Y-3)/2) + 4
        d4 = (math.sin((X - 6)/2)/2) + (math.sin((Y-6)/2)/2) + 2
        # print(f"True val = {d2}, {d3}, {d4}")
        data_array = np.hstack((d2,d3, d4)).reshape((-1,1))
        timestamps = timestamps_array
        data_list = data_array
        trace = Trace(timestamps, data_list)
        return BasicResult(trace)


model = NLFModel()
 """

""" model = NLFModel()

initial_conditions = [
    np.array([-5,5]),
    np.array([-5,5]),
]


options = Options(runs=1, iterations=5, interval=(0, 1),  static_parameters=initial_conditions ,signals=[])


phi_2 = "x>=0"
phi_3 = "y>=2"
phi_4 = "z>=1"

fn_list_1 = [phi_2, phi_3, phi_4]
pred_map_1 = {"x": ([0,1], 0), "y":([0, 1],1), "z":([0, 1], 2)} """

class NLFModel(Model[NLFResultT, None]):
    def simulate(
        self, static: ModelInputs, intrvl: Interval
    ) -> NLFResultT:
        print(static)
        timestamps_array = np.array(1.0).flatten()
        X = static.static[0]
        Y = static.static[1]
        """d0=3 * math.sin(1 * x + 0.25) + 2 * math.sin(0.25 * y + 1) + 5.55829449361776 +0.0000001
        d1=1 * math.sin(1 * x + 0.25) + 3 * math.sin(1 * y + 0.0) + 4.8475111142115015 +0.0000001
        d2=2 * math.sin(0.75 * x + 0.75) + 1 * math.sin(1 * y + 0.75) + 3.3413293823480505 +0.0000001
        d3=2 * math.sin(0.75 * x + 0.25) + 3 * math.sin(1 * y + 0.75) + 6.92809455383691 +0.0000001
        d4=1 * math.sin(0.5 * x + 0.25) + 2 * math.sin(0.75 * y + 0.75) + 4.7817556855111825 +0.0000001
        d5=2 * math.sin(0.25 * x + 0.25) + 2 * math.sin(0.25 * y + 1) + 5.885213609860813 +0.0000001
        d6=2 * math.sin(0.75 * x + 1) + 1 * math.sin(1 * y + 1) + 3.4809572543750407 +0.0000001
        d7=1 * math.sin(0.25 * x + 0.75) + 3 * math.sin(1 * y + 0.25) + 4.202622459983928 +0.0000001
        d8=2 * math.sin(0.25 * x + 0.0) + 1 * math.sin(0.25 * y + 0.0) + 4.216247564296491 +0.0000001
        d9=3 * math.sin(0.25 * x + 0.25) + 3 * math.sin(0.25 * y + 0.5) + 6.506660735841987 +0.0000001
        d10=1 * math.sin(0.75 * x + 0.75) + 2 * math.sin(0.5 * y + 0.0) + 3.6527634743045203 +0.0000001
        d11=1 * math.sin(1 * x + 1) + 1 * math.sin(0.25 * y + 0.0) + 3.368380707245416 +0.0000001
        d12=3 * math.sin(1 * x + 0.5) + 1 * math.sin(0.25 * y + 0.0) + 4.399288322747687 +0.0000001
        d13=3 * math.sin(0.75 * x + 1) + 2 * math.sin(0.75 * y + 0.5) + 5.8802393811963345 +0.0000001
        d14=1 * math.sin(1 * x + 0.25) + 1 * math.sin(0.25 * y + 0.75) + 3.266296637632359 +0.0000001
        d15=2 * math.sin(0.5 * x + 1) + 3 * math.sin(0.5 * y + 1) + 5.013442721250685 +0.0000001
        d16=1 * math.sin(0.25 * x + 0.75) + 1 * math.sin(0.5 * y + 0.5) + 3.2164685480268127 +0.0000001
        d17=1 * math.sin(1 * x + 0.0) + 2 * math.sin(1 * y + 0.5) + 4.757407820716709 +0.0000001
        d18=2 * math.sin(0.5 * x + 0.0) + 1 * math.sin(0.75 * y + 0.5) + 3.631979865651728 +0.0000001
        d19=3 * math.sin(0.5 * x + 0.0) + 3 * math.sin(0.75 * y + 0.5) + 7.5801720095763905 +0.0000001
        d20=2 * math.sin(1 * x + 0.75) + 3 * math.sin(0.75 * y + 0.5) + 5.889850805659178 +0.0000001
        d21=2 * math.sin(0.5 * x + 0.0) + 3 * math.sin(0.5 * y + 0.25) + 6.426426341968451 +0.0000001
        d22=1 * math.sin(0.5 * x + 0.5) + 1 * math.sin(0.5 * y + 0.0) + 3.2260759244078865 +0.0000001
        d23=1 * math.sin(0.5 * x + 0.0) + 1 * math.sin(0.25 * y + 0.25) + 2.2448767389046007 +0.0000001
        d24=1 * math.sin(0.25 * x + 0.25) + 2 * math.sin(0.25 * y + 0.0) + 3.1586622177640145+0.0000001
        d25=2 * math.sin(0.5 * x + 0.5) + 1 * math.sin(0.25 * y + 1) + 3.4556338022775086 +0.0000001
        d26=1 * math.sin(0.25 * x + 1) + 3 * math.sin(0.25 * y + 0.75) + 4.694123414840719 +0.0000001
        d27=1 * math.sin(1 * x + 0.0) + 1 * math.sin(0.75 * y + 0.75) + 3.8646205882050144 +0.0000001
        d28=1 * math.sin(0.5 * x + 0.25) + 2 * math.sin(0.5 * y + 0.5) + 3.3367664122811633+0.0000001 
        d29=3 * math.sin(0.5 * x + 0.25) + 1 * math.sin(1 * y + 0.75) + 3.999990845156247 + 0.0000001"""

        """d0=0.5023406427542574 * math.sin(0.5912539997695563 * x + 2.808372553919802) + 0.17056012434473922 * math.sin(0.041948123357664735 * y + 1.740117585089841) + 1.353159762861348 - 0.10518398517617111
        d1=0.6388841493194803 * math.sin(0.8776898572351975 * x + 2.1512734057027254) + 0.6746720979817925 * math.sin(0.7644456525282709 * y + 1.5406297643919233) + 2.071005360052174 - 0.10518398517617111
        d2=0.27485761238566775 * math.sin(0.9777313008611347 * x + 2.8262244373128502) + 0.8098325969590967 * math.sin(0.6860446102380305 * y + 1.2691293807455615) + 1.3359575902322138 - 0.10518398517617111
        d3=0.41884403923280944 * math.sin(0.23997920645944693 * x + 1.682581210214911) + 0.4965112014187367 * math.sin(0.9716082687762403 * y + 0.6676810475965664) + 1.1402965001539684 - 0.10518398517617111
        d4=0.8362608275194153 * math.sin(0.7174069945479022 * x + 0.8588008371019411) + 0.7958650400342457 * math.sin(0.8984388346133757 * y + 1.225353478513082) + 3.2147984698601677 - 0.10518398517617111
        d5=0.9759345014125009 * math.sin(0.7975888555939373 * x + 2.8338242230125172) + 0.7891931872203738 * math.sin(0.950167796788191 * y + 1.01598513400449) + 2.36283205394734 - 0.10518398517617111
        d6=0.5032227915815926 * math.sin(0.10486537502281557 * x + 1.9542868248275502) + 0.12224638522089781 * math.sin(0.8384545589236352 * y + 1.620486994119748) + 0.7536497088558339 - 0.10518398517617111
        d7=0.4666390457727949 * math.sin(0.4363641805254014 * x + 0.7656479058281769) + 0.2589852095195313 * math.sin(0.5249691813598234 * y + 3.0823431838577213) + 0.7200361300833482 - 0.10518398517617111
        d8=0.21435226490500459 * math.sin(0.509560815585475 * x + 0.2190941477931356) + 0.02597837608817788 * math.sin(0.5312828398997534 * y + 0.267498600726103) + 1.5664090535637665 - 0.10518398517617111
        d9=0.7239651829461274 * math.sin(0.32787738189765325 * x + 1.7729208382618784) + 0.2708041591803837 * math.sin(0.4978882482695808 * y + 2.745497628322427) + 1.1862575668631066 - 0.10518398517617111
        d10=0.07767484585898166 * math.sin(0.3698479236668407 * x + 1.2696594286128293) + 0.1523465944800465 * math.sin(0.518410929427203 * y + 2.234871744761967) + 1.5083401461238428 - 0.10518398517617111
        d11=0.6684921507391868 * math.sin(0.5302549038125882 * x + 0.9886525139021936) + 0.0505300311476109 * math.sin(0.9583522949790734 * y + 2.844425541318538) + 1.028248503173883 - 0.10518398517617111
        d12=0.8529237636966653 * math.sin(0.3316288107625617 * x + 1.2107534603896573) + 0.316251585576644 * math.sin(0.44768068245595194 * y + 0.6256120946553954) + 1.8810702842846032 - 0.10518398517617111
        d13=0.1052216067902032 * math.sin(0.0951179822181153 * x + 2.012847663499907) + 0.9493172414836285 * math.sin(0.9157882198501728 * y + 2.889085262146066) + 1.9812379213586935 - 0.10518398517617111
        d14=0.7035113809716379 * math.sin(0.5552332551351364 * x + 2.8298856266562114) + 0.006423242731325218 * math.sin(0.42608296266184553 * y + 0.6813368511797492) + 1.8363411702691739 - 0.10518398517617111
        d15=0.9097206234523215 * math.sin(0.46719536204956447 * x + 0.8381922658869642) + 0.2631246960629887 * math.sin(0.5815924359918069 * y + 2.5371681100600614) + 2.578002323414588 - 0.10518398517617111
        d16=0.11971215419983539 * math.sin(0.9514809607704563 * x + 2.0033216157741465) + 0.32355092468364866 * math.sin(0.710919329579579 * y + 0.03877991299569011) + 0.965709153770193 - 0.10518398517617111
        d17=0.7976469966459802 * math.sin(0.3364855921770272 * x + 1.4640788279839552) + 0.5386904833607042 * math.sin(0.705675338849909 * y + 2.989361392586617) + 2.609539922148763 - 0.10518398517617111
        d18=0.3080508593655108 * math.sin(0.8458505448003699 * x + 3.1019967319881636) + 0.8250458025362863 * math.sin(0.6415618495134483 * y + 0.5505412070267844) + 2.4694610024096475 - 0.10518398517617111
        d19=0.03150539489451254 * math.sin(0.18347500537198347 * x + 0.40445414933742146) + 0.1619618305520023 * math.sin(0.05891263202279018 * y + 1.9268692321332757) + 2.0678216645678678 - 0.10518398517617111
        d20=0.770329218632887 * math.sin(0.233547717615173 * x + 0.10591332897934619) + 0.1343545178323936 * math.sin(0.916490741607739 * y + 0.1469552670842519) + 2.1149627898396184 - 0.10518398517617111
        d21=0.41929162362091366 * math.sin(0.06664372300276233 * x + 1.0758133228679196) + 0.19292066967148436 * math.sin(0.6790551265770117 * y + 0.07793364980016083) + 1.6567944736614804 - 0.10518398517617111
        d22=0.019422591647582865 * math.sin(0.0006531817047235045 * x + 1.828387689035127) + 0.11942298574144394 * math.sin(0.7921154163566106 * y + 1.1140800668484019) + 0.20280634933937425 - 0.10518398517617111
        d23=0.3906975903686123 * math.sin(0.9573061772487119 * x + 0.44603602973888523) + 0.41234452977680625 * math.sin(0.6140287007158612 * y + 1.9137664029625214) + 2.4294297764674795 - 0.10518398517617111
        d24=0.885891979308114 * math.sin(0.9845820977095348 * x + 1.506472135921346) + 0.722432056183763 * math.sin(0.2822384772892609 * y + 1.602598429339518) + 2.97311452593952 - 0.10518398517617111
        d25=0.13232562692991223 * math.sin(0.38911740797186745 * x + 0.023739176079329266) + 0.2766071946718739 * math.sin(0.463531777844425 * y + 0.7047599067435683) + 1.0492756782754369 - 0.10518398517617111
        d26=0.5541018694010074 * math.sin(0.01692423532343157 * x + 1.320013314866821) + 0.9520159024149156 * math.sin(0.4079182306272702 * y + 1.3479562902540387) + 2.699423166868812 - 0.10518398517617111
        d27=0.8671631581430898 * math.sin(0.4043279228576059 * x + 1.0238733207325836) + 0.5929092033760791 * math.sin(0.48897890332255056 * y + 0.0926684440802445) + 1.5445343162121623 - 0.10518398517617111
        d28=0.10464954473726429 * math.sin(0.9336100387845061 * x + 1.793798824026412) + 0.8425567373459573 * math.sin(0.04741457673510452 * y + 2.324554656664999) + 2.3635455396022165 - 0.10518398517617111
        d29=0.587921252815029 * math.sin(0.9333227093359974 * x + 0.7393353149694241) + 0.8710037872184473 * math.sin(0.5723609311088781 * y + 1.180539286285457) + 2.5230433993082357 - 0.10518398517617111"""

        """d0=0.410271934296341 * math.sin(0.31749166137974444 * x + 3.414235551061579) + 1.7652959525445269 * math.sin(0.38688344598162405 * y + 4.129915202504272) + 2.635311938687326 
        d1=0.551590301962013 * math.sin(0.3380695707915824 * x + 2.99209733441971) + 0.9919108116224034 * math.sin(0.2737700940482345 * y + 3.016138653872876) + 1.8704615372866784 - 0.025375643039574403
        d2=0.05990080781425822 * math.sin(0.44275068365253756 * x + 5.757693083193321) + 0.778184646708369 * math.sin(0.363377557436778 * y + 4.020695129363184) + 1.8236651880837238 
        d3=0.5897321512386051 * math.sin(0.25636114680961314 * x + 1.9157893377362838) + 1.8404192204778744 * math.sin(0.424102626544859 * y + 0.830011929919672) + 4.0068867927758465 
        d4=0.8223076573452401 * math.sin(0.8434046930577452 * x + 2.7214154818652054) + 1.882234487202133 * math.sin(0.8458913723930879 * y + 1.0796787773296896) + 4.514683469293758 
        d5=1.1117125547265323 * math.sin(0.7195493640351082 * x + 3.448458877154165) + 1.3545887318849181 * math.sin(0.33121683625039533 * y + 0.2723529285052347) + 3.1082879656581093 
        d6=1.9108254755233531 * math.sin(0.7002518093002594 * x + 4.198332307424629) + 0.6018687558634548 * math.sin(0.176228215230752 * y + 2.6704858402561884) + 2.963631420404611 
        d7=1.5631124735222683 * math.sin(0.13178378469999208 * x + 6.019395120251168) + 1.5437122959590623 * math.sin(0.2167631278555623 * y + 5.118061301015436) + 4.746817757556984 
        d8=0.010983824510094164 * math.sin(0.9420798155833985 * x + 1.197946974053397) + 0.6111326800819104 * math.sin(0.8162933820943148 * y + 2.846667561333681) + 1.6274951753103528 
        d9=0.44352332377162207 * math.sin(0.4761675845607003 * x + 3.1202794303059425) + 0.6511660208598742 * math.sin(0.13971233535062455 * y + 3.1962568041017407) + 2.193711270286192 
        d10=1.0705027360066373 * math.sin(0.23056140600001235 * x + 3.248070012048077) + 1.1616672550162062 * math.sin(0.5729519662925425 * y + 1.4794676439491672) + 2.7321132623844395 
        d11=1.351640861799398 * math.sin(0.7210293384241784 * x + 4.028055506572028) + 1.4270135084569144 * math.sin(0.7373227962783412 * y + 4.1368064938547455) + 4.505194208260732 
        d12=0.7298794202893342 * math.sin(0.3543248763864394 * x + 0.2389865315708707) + 0.5875571187604235 * math.sin(0.11313023877535927 * y + 3.630917286751547) + 3.068507447178899 
        d13=0.08670537029051184 * math.sin(0.28580625101870594 * x + 0.563460093390083) + 1.5592702006937111 * math.sin(0.6904101851646978 * y + 2.6274985942471605) + 3.3860458280728256 
        d14=1.9596337418111929 * math.sin(0.9672582930281731 * x + 4.916463761911229) + 1.4757097407119275 * math.sin(0.6259276219369354 * y + 2.2197324794841222) + 5.415200651933477 
        d15=0.47860292499482515 * math.sin(0.9913211021148541 * x + 4.917810610108077) + 1.375042819343043 * math.sin(0.5020755100146238 * y + 6.223380180842337) + 2.923457968695555 
        d16=0.2963322680660594 * math.sin(0.8329519831063653 * x + 2.5730215096821274) + 0.9195156771528439 * math.sin(0.47063108224391903 * y + 5.39491664687897) + 1.942026675435496 
        d17=0.975433313136193 * math.sin(0.24640187096364957 * x + 1.6721749410054134) + 1.5564150402685262 * math.sin(0.7919923292695057 * y + 2.672495661754594) + 2.974054711629686 
        d18=1.1976970601085197 * math.sin(0.2535650404346114 * x + 5.0806035397424445) + 1.8492401914005916 * math.sin(0.8064376383445838 * y + 0.6215156215348789) + 4.017891022168048 
        d19=0.17111526737740235 * math.sin(0.40523395386018934 * x + 0.7114077252680748) + 0.6799910729914458 * math.sin(0.717201743330617 * y + 0.8112119384956692) + 2.242733975568779 
        d20=1.725591689939213 * math.sin(0.10074645264254184 * x + 3.289283118361136) + 1.6259516511913596 * math.sin(0.9113159534944419 * y + 5.053768220973497) + 4.576296803799822 
        d21=1.2076935964602935 * math.sin(0.2360033356949965 * x + 2.6752818875304114) + 0.6607605948412593 * math.sin(0.9050624444048497 * y + 0.8867644572455392) + 3.8013554372551077 
        d22=1.348934328870659 * math.sin(0.5059200187823025 * x + 4.14791349227088) + 1.4880019484435967 * math.sin(0.45716287781439735 * y + 0.5397897986704212) + 3.225025970475852 
        #d25=0.3379671004884286 * math.sin(0.8373723295679552 * x + 2.9910395786107924) + 0.8424788910490754 * math.sin(0.10357191069019747 * y + 2.4947568513276326) + 1.1833000836255878- 0.3880897062838051
        d23=1.5117007466529944 * math.sin(0.5059200187823025 * x + 4.14791349227088) + 1.4880019484435967 * math.sin(0.45716287781439735 * y + 0.5397897986704212) + 5.0291798087891135-  2.0041014709029863 - 0.025375643039574403 
        d24=1.4619753081420452 * math.sin(0.1805309619509498 * x + 4.275881305187563) + 1.8699948961982513 * math.sin(0.7092073193979879 * y + 4.272154791726991) + 3.9517890899213852 
        d25=1.915421115083869 * math.sin(0.22134981276975763 * x + 1.1444643853161336) + 0.806961197038881 * math.sin(0.5502976413122382 * y + 3.2301782124087994) + 3.844945670258518 
        d26=1.1307273670482798 * math.sin(0.6970709389832681 * x + 2.7614872395805774) + 0.7464557529908152 * math.sin(0.11880246405434443 * y + 1.782194234736948) + 3.6393937650008383 
        d27=1.6249552751113483 * math.sin(0.14620330753420935 * x + 0.7999729150510807) + 1.796135923946966 * math.sin(0.72929665506175 * y + 2.181340853279292) + 3.597146548916208 
        d28=1.847381876976197 * math.sin(0.5828970590750181 * x + 5.792678038831498) + 0.6664979868818186 * math.sin(0.48030231113139954 * y + 4.907387141187375) + 3.6396316457344224 
        d29=0.6699808998061616 * math.sin(0.20290273335520584 * x + 1.2944774129102508) + 1.3996576948029964 * math.sin(0.1751097769559792 * y + 1.0681679863831435) + 2.3101839154266397""" 
        
       
        d0 = 0.8077039507222558 * math.sin(0.31749166137974444 * X + 3.414235551061579) + 1.7652959525445269 * math.sin(0.38688344598162405 * Y + 4.129915202504272) + 8.208314357501266-  2.0041014709029863 - 0.025375643039574403
        d1 = 0.9136927264715098 * math.sin(0.3380695707915824 * X + 2.99209733441971) + 0.9919108116224034 * math.sin(0.2737700940482345 * Y + 3.016138653872876) + 4.876355287140777-  2.0041014709029863 - 0.025375643039574403
        d2 = 0.5449256058606937 * math.sin(0.44275068365253756 * X + 5.757693083193321) + 0.778184646708369 * math.sin(0.363377557436778 * Y + 4.020695129363184) + 6.522903561787862-  2.0041014709029863 - 0.025375643039574403
        d3 = 0.9422991134289538 * math.sin(0.25636114680961314 * X + 1.9157893377362838) + 1.8404192204778744 * math.sin(0.424102626544859 * Y + 0.830011929919672) + 8.000792780533539-  2.0041014709029863 - 0.025375643039574403
        d4 = 1.1167307430089302 * math.sin(0.8434046930577452 * X + 2.7214154818652054) + 1.882234487202133 * math.sin(0.8458913723930879 * Y + 1.0796787773296896) + 8.584307539751084-  2.0041014709029863 - 0.025375643039574403
        d5 = 1.3337844160448993 * math.sin(0.7195493640351082 * X + 3.448458877154165) + 1.3545887318849181 * math.sin(0.33121683625039533 * Y + 0.2723529285052347) + 5.663920925501769-  2.0041014709029863 - 0.025375643039574403
        d6 = 1.9331191066425149 * math.sin(0.7002518093002594 * X + 4.198332307424629) + 0.6018687558634548 * math.sin(0.176228215230752 * Y + 2.6704858402561884) + 5.186297200429628-  2.0041014709029863 - 0.025375643039574403
        d7 = 1.6723343551417011 * math.sin(0.13178378469999208 * X + 6.019395120251168) + 1.5437122959590623 * math.sin(0.2167631278555623 * Y + 5.118061301015436) + 8.158936698074255-  2.0041014709029863 - 0.025375643039574403
        d8 = 0.5082378683825706 * math.sin(0.9420798155833985 * X + 1.197946974053397) + 0.6111326800819104 * math.sin(0.8162933820943148 * Y + 2.846667561333681) + 6.5724009046809915-  2.0041014709029863 - 0.025375643039574403
        d9 = 0.8326424928287166 * math.sin(0.4761675845607003 * X + 3.1202794303059425) + 0.6511660208598742 * math.sin(0.13971233535062455 * Y + 3.1962568041017407) + 6.80650904202186-  2.0041014709029863 - 0.025375643039574403
        d10 = 1.302877052004978 * math.sin(0.23056140600001235 * X + 3.248070012048077) + 1.1616672550162062 * math.sin(0.5729519662925425 * Y + 1.4794676439491672) + 5.308812406289112-  2.0041014709029863 - 0.025375643039574403
        d11 = 1.5137306463495486 * math.sin(0.7210293384241784 * X + 4.028055506572028) + 1.4270135084569144 * math.sin(0.7373227962783412 * Y + 4.1368064938547455) + 8.375303822896168-  2.0041014709029863 - 0.025375643039574403
        d12 = 1.0474095652170006 * math.sin(0.3543248763864394 * X + 0.2389865315708707) + 0.5875571187604235 * math.sin(0.11313023877535927 * Y + 3.630917286751547) + 8.436631498207975-  2.0041014709029863 - 0.025375643039574403
        d13 = 0.5650290277178839 * math.sin(0.28580625101870594 * X + 0.563460093390083) + 1.5592702006937111 * math.sin(0.6904101851646978 * Y + 2.6274985942471605) + 8.409129870606627-  2.0041014709029863 - 0.025375643039574403
        d14 = 1.9697253063583946 * math.sin(0.9672582930281731 * X + 4.916463761911229) + 1.4757097407119275 * math.sin(0.6259276219369354 * Y + 2.2197324794841222) + 9.008597151411013-  2.0041014709029863 - 0.025375643039574403
        d15 = 0.8589521937461189 * math.sin(0.9913211021148541 * X + 4.917810610108077) + 1.375042819343043 * math.sin(0.5020755100146238 * Y + 6.223380180842337) + 6.733484788779338-  2.0041014709029863 - 0.025375643039574403
        d16 = 0.7222492010495445 * math.sin(0.8329519831063653 * X + 2.5730215096821274) + 0.9195156771528439 * math.sin(0.47063108224391903 * Y + 5.39491664687897) + 5.8744010534266025-  2.0041014709029863 - 0.025375643039574403
        d17 = 1.2315749848521447 * math.sin(0.24640187096364957 * X + 1.6721749410054134) + 1.5564150402685262 * math.sin(0.7919923292695057 * Y + 2.672495661754594) + 5.164470123447538-  2.0041014709029863 - 0.025375643039574403
        d18 = 1.3982727950813898 * math.sin(0.2535650404346114 * X + 5.0806035397424445) + 1.8492401914005916 * math.sin(0.8064376383445838 * Y + 0.6215156215348789) + 6.486338654532464-  2.0041014709029863 - 0.025375643039574403
        d19 = 0.6283364505330518 * math.sin(0.40523395386018934 * X + 0.7114077252680748) + 0.6799910729914458 * math.sin(0.717201743330617 * Y + 0.8112119384956692) + 7.538023315884949-  2.0041014709029863 - 0.025375643039574403
        d20 = 1.7941937674544097 * math.sin(0.10074645264254184 * X + 3.289283118361136) + 1.6259516511913596 * math.sin(0.9113159534944419 * Y + 5.053768220973497) + 7.120837884558247-  2.0041014709029863 - 0.025375643039574403
        d21 = 1.4057701973452201 * math.sin(0.2360033356949965 * X + 2.6752818875304114) + 0.6607605948412593 * math.sin(0.9050624444048497 * Y + 0.8867644572455392) + 8.89120734276901-  2.0041014709029863 - 0.025375643039574403
        d22 = 1.5117007466529944 * math.sin(0.5059200187823025 * X + 4.14791349227088) + 1.4880019484435967 * math.sin(0.45716287781439735 * Y + 0.5397897986704212) + 5.0291798087891135-  2.0041014709029863 - 0.025375643039574403
        d23 = 0.7534753253663214 * math.sin(0.8373723295679552 * X + 2.9910395786107924) + 0.8424788910490754 * math.sin(0.10357191069019747 * Y + 2.4947568513276326) + 4.066089458105331-  2.0041014709029863 - 0.025375643039574403
        d24 = 1.596481481106534 * math.sin(0.1805309619509498 * X + 4.275881305187563) + 1.8699948961982513 * math.sin(0.7092073193979879 * Y + 4.272154791726991) + 8.608501441837843-  2.0041014709029863 - 0.025375643039574403
        d25 = 1.9365658363129017 * math.sin(0.22134981276975763 * X + 1.1444643853161336) + 0.806961197038881 * math.sin(0.5502976413122382 * Y + 3.2301782124087994) + 6.8653626232245415-  2.0041014709029863 - 0.025375643039574403
        d26 = 1.3480455252862098 * math.sin(0.6970709389832681 * X + 2.7614872395805774) + 0.7464557529908152 * math.sin(0.11880246405434443 * Y + 1.782194234736948) + 8.464480840289479-  2.0041014709029863 - 0.025375643039574403
        d27 = 1.7187164563335111 * math.sin(0.14620330753420935 * X + 0.7999729150510807) + 1.796135923946966 * math.sin(0.72929665506175 * Y + 2.181340853279292) + 4.499092602529856-  2.0041014709029863 - 0.025375643039574403
        d28 = 1.8855364077321477 * math.sin(0.5828970590750181 * X + 5.792678038831498) + 0.6664979868818186 * math.sin(0.48030231113139954 * Y + 4.907387141187375) + 6.873333682576138-  2.0041014709029863 - 0.025375643039574403
        d29 = 1.0024856748546211 * math.sin(0.20290273335520584 * X + 1.2944774129102508) + 1.3996576948029964 * math.sin(0.1751097769559792 * Y + 1.0681679863831435) + 4.660317529928825-  2.0041014709029863 - 0.025375643039574403


       
        # print(f"True val = {d2}, {d3}, {d4}")
        data_array = np.hstack((d0,d1,d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29)).reshape((-1,1))
        timestamps = timestamps_array
        data_list = data_array
        trace = Trace(timestamps, data_list)
        return BasicResult(trace)




model = NLFModel()

initial_conditions = [
    np.array([-5,5]),
    np.array([-5,5]),
]


options = Options(runs=1, iterations=5, interval=(0, 1),  static_parameters=initial_conditions ,signals=[])

phi_2 = "a>=0"
phi_3 = "b>=0"
phi_4 = "c>=0"
phi_5 = "d>=0"
phi_6 = "e>=0"
phi_7 = "f>=0"
phi_8 = "g>=0"
phi_9 = "h>=0"
phi_10 = "i>=0"
phi_11 = "j>=0"
phi_12 = "k>=0"
phi_13 = "l>=0"
phi_14 = "m>=0"
phi_15 = "n>=0"
phi_16 = "o>=0"
phi_17 = "p>=0"
phi_18 = "q>=0"
phi_19 = "r>=0"
phi_20 = "s1>=0"
phi_21 = "t>=0"
phi_22 = "u>=0"
phi_23 = "v>=0"
phi_24 = "x>=0"
phi_25 = "y>=0"
phi_26 = "z>=0"
phi_27 = "P>= 0"
phi_28 = "Q>=0"
phi_29 = "R>=0"
phi_30 = "S1>=0"
phi_31 = "T>=0"

fn_list_1 = [phi_2, phi_3, phi_4,phi_5, phi_6 ,phi_7 ,phi_8, phi_9 ,phi_10,phi_11,phi_12,phi_13, phi_14, phi_15, phi_16, phi_17, phi_18, phi_19,phi_20, phi_21, phi_22, phi_23, phi_24, phi_25, phi_26, phi_27, phi_28, phi_29,phi_30, phi_31]
 # Change this
pred_map_1 = {"a": ([0,1], 0),
              "b": ([0,1], 1),
              "c": ([0,1], 2),
              "d": ([0,1], 3),
              "e": ([0,1], 4),
              "f": ([0,1], 5),
              "g": ([0,1], 6),
              "h": ([0,1], 7),
              "i": ([0,1], 8),
              "j": ([0,1], 9),
              "k": ([0,1], 10),
              "l": ([0,1], 11),
              "m": ([0,1], 12),
              "n": ([0,1], 13),
              "o": ([0,1], 14),
              "p": ([0,1], 15),
              "q": ([0,1], 16),
              "r": ([0,1], 17),
              "s1": ([0,1], 18),
              "t": ([0,1], 19),
              "u": ([0,1], 20),
              "v": ([0,1], 21),
              "x": ([0,1], 22),
              "y": ([0,1], 23),
              "z": ([0,1], 24),
              "P": ([0,1], 25),
              "Q": ([0,1], 26),
              "R": ([0,1], 27),
              "S1": ([0,1], 28),
              "T": ([0,1], 29)}


is_budget = 20
max_budget = 30
cs_budget = 1000
spec_list = [fn_list_1]
predicate_mapping = pred_map_1
region_support = np.array([[-5., 5.], [-5., 5.]])
tf_dim = 2
R = 20
M = 500


top_k = 3
Benchmark_name = "NLF_trial"
#UNCOMMENT THE PICKLE LINe	
seed = 123457

total_runs = 1
from lsemibo.coreAlgorithm.specification import Requirement
specification = Requirement(tf_dim, fn_list_1, pred_map_1)


for i in range(total_runs):

    optimizer = LSemiBOOptimizer( 
        method = "falsification_elimination",
        is_budget = is_budget,
        max_budget= max_budget,
        cs_budget = cs_budget,
        top_k = top_k,
        classified_sample_bias = 1,
        tf_dim = tf_dim,
        R = R,  
        M = M,
        gpr_model = InternalGPR(),
        classifier_model = InternalClassifier(),
        is_type = "lhs_sampling",
        cs_type= "lhs_sampling",
        pi_type= "lhs_sampling",
        seed= seed+i)

    result = staliro(model, specification, optimizer, options)
    with open(f'NLF_{is_budget}_{max_budget}_seed_{seed+i}.pkl', 'wb') as file:
        pickle.dump(result, file)

with open(f'NLF_{is_budget}_{max_budget}_seed_{seed+i}.pkl', 'rb') as f:
    data = pickle.load(f)

# print([x for x in data.runs[0].model_timing.durations])

print(data.runs[0].result.start_timestamp)
print(data.runs[0].result.iteration_timestamps)
x = np.array([data.runs[0].result.start_timestamp] + data.runs[0].result.iteration_timestamps)

print(np.diff(x))


#print(result)
# for runs in range(total_runs):
    
#     lsemibo = LSemiBO(Benchmark_name, runs, is_budget, max_budget, cs_budget, top_k, 0.8, model, spec_list, predicate_mapping, tf_dim, options, R, M, is_type = "lhs_sampling", cs_type = "lhs_sampling", seed = 12345)
#     x_train, y_train, time_taken = lsemibo.sample(InternalGPR(), InternalClassifier())

#     print(x_train)
#     print(y_train)
#     print(time_taken)