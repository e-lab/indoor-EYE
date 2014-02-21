-------------------------------------------------------------------------------
-- Define indoor classes of imagenet dataset
-- Artem Kuharenko
-------------------------------------------------------------------------------

--{class name, imagenet class ids}
if opt.subsample_name == 'elab' then

   classes = {{'computer-mouse',  {511}},
              {'printer',         {556}},
              {'cellphone',       {914}},
              {'cup',             {859}},
              {'laptop',          {228}},
              {'keyboard',        {543}},
              {'desk',            {313}},
              {'bottle-of-water', {958}},
              {'trash-can',       {752}}
             }

elseif opt.subsample_name == 'kitchen' then

   classes = {{'cup',           {859}},
              {'beer',          {777}},
              {'dinning-table', {315}},
              {'apple',         {318}},
              {'orange',        {319}},
              {'banana',        {323}},
              {'cockscrew',     {376}},
              {'can-opener',    {377}},
              {'stove',         {516}},
              {'wall-clock',    {524}},
              {'microwave',     {661}},
              {'dutch-oven',    {662}},
              {'toaster',       {664}},
              {'dishwasher',    {667}},
              {'refrigerator',  {668}},
              {'washer',        {669}},
              {'crock-pot',     {670}},
              {'frying-pan',    {671}},
              {'wok',           {672}},
              {'coffee-pot',    {674}},
              {'tee-pot',       {675}},
              {'plate-rack',    {731}},
              {'plate',         {754}},
              {'pop-bottle',    {788}},
              {'water-jug',     {819}},
              {'soup-bowl',     {822}},
              {'mixing-bowl',   {829}},
              {'saltshaker',    {952}},
              {'goblet',        {955}},
--              {'laptop',        {228}},
--              {'computer',      {550}},
--              {'notebook',      {552}},
              {'coffe-mug',     {996}}
             }

elseif opt.subsample_name == 'bedroom' then

   classes = {{'chiffonier',    {303}},
              {'table lamp',    {304}},
              {'file',          {305}},
              {'folding-chair', {309}},
              {'rocking-chair', {310}},
              {'studio-coach',  {311}},
              {'desk',          {313}},
              {'wardrobe',      {317}},
              {'loud-speaker',  {508}},
              {'screen',        {510}},
              {'heater',        {515}},
              {'keyboard',      {543}},
              {'radiator',      {571}},
              {'lamp-shade',    {814}},
              {'monitor',       {869}},
              {'pillow',        {888}},
              {'quilt',         {976}},
              {'handkerchief',  {750}}
             }


elseif opt.subsample_name == 'indoor51' then

   classes = {{'domestic-animal',    {2, 3, 4, 5, 8, 10, 15, 17, 18, 19, 20, 21, 25, 26, 27, 29, 31, 32, 33, 36, 40, 41, 42, 43, 45, 46, 47, 49, 50, 51, 56, 59, 60, 63, 64, 66, 68, 69, 70, 71, 72, 77, 79, 82, 84, 86, 87, 88, 89, 90, 91, 93, 94, 95, 97, 98, 99, 101, 105, 106, 107, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 139, 140, 141, 143, 144, 145, 146, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 160, 161, 168, 170, 171, 172, 173, 174, 176, 177, 179, 180, 181, 184, 187, 189, 191, 192, 196, 197, 198, 200, 204, 207, 208, 210, 211}},
              {'accessory',          {220, 370, 376, 377, 583, 584, 835, 928, 939}},
              {'musical-instrument', {223, 227, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 575, 799}},
              {'food',               {229, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 331, 332, 613, 616, 617, 663, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 754, 768, 771, 793, 806, 813, 822, 823, 830, 844, 864, 873, 885, 900, 948, 953, 968, 974, 975, 993, 999}},
              {'bathroom',           {312, 378, 765, 877, 884, 889, 906, 960}},
              {'electronics',        {509, 511, 534, 556, 560, 573, 578, 764, 789, 914, 980}},
              {'tools',              {369, 372, 373, 374, 375, 379, 380, 382, 519, 585, 587, 891}},
              {'writing-instrument', {842, 850, 907, 934, 997}},
              {'kitchenware',        {514, 671, 676, 761, 824, 829, 892, 951, 952}},
              {'clock',              {522, 523, 524, 525, 528, 529}},
              {'switch',             {561}},
              {'light',              {304, 546, 591, 592, 593, 594, 814}},
              {'curtain',            {747}},
              {'clothes',            {748, 757, 759, 770, 794, 797, 801, 802, 805, 815, 817, 825, 836, 837, 845, 853, 854, 855, 866, 871, 872, 880, 881, 896, 911, 925, 935, 937, 938, 941, 961, 986, 998}},
              {'box',                {749, 762, 898}},
              {'shoes',              {751, 760, 910, 973, 979}},
              {'jewellery',          {755}},
              {'rug',                {769, 972}},
              {'book',               {774, 930}},
              {'toy',                {786, 931}},
              {'table-game',         {791, 963}},
              {'beverage',           {777, 788, 798, 811, 831, 955}},
              {'cosmetics',          {808, 810, 867, 883, 894, 895}},
              {'bag',                {818, 847}},
              {'broom',              {828, 851}},
              {'flower-pot',         {838, 874}},
              {'cup',                {859, 947, 996}},
              {'drug',               {901}},
              {'window-shade',       {904}},
              {'bed',                {299, 311, 976}},
              {'computer',           {228, 543, 550, 551, 552, 553}},
              {'music-player',       {508, 863, 929, 978, 987}},
              {'camera',             {597, 857, 965, 988}},
              {'screen',             {510, 696, 869, 944}},
              {'pot',                {662, 670, 672, 673}},
              {'jar',                {674, 675, 772, 819, 946, 983, 991}},
              {'fan',                {512}},
              {'heater',             {515}},
              {'child-bed',          {296, 297, 298}},
              {'chair',              {307, 308, 309, 310}},
              {'closet',             {300, 301, 302, 303, 305, 317, 832}},
              {'table',              {313, 314, 315}},
              {'toaster',            {664}},
              {'stove',              {516}},
              {'microwave',          {661}},
              {'dishwasher',         {667}},
              {'refrigerator',       {668}},
              {'washer',             {669}},
              {'hand-blower',        {505}},
              {'iron',               {659}},
              {'vacuum',             {666}}
             }
end

class_names = {}
for i = 1, #classes do
   class_names[i] = classes[i][1]
end
