import os
import numpy as np
import requests
import argparse
import json
import time
import logging
import csv

from multiprocessing import  Process, Value, Lock
from multiprocessing.pool import ThreadPool as Pool

from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL


def main(args):
    logging.basicConfig(filename='imagenet_scarper.log', level=logging.INFO)
    if args.debug:
        logging.basicConfig(filename='imagenet_scarper.log', level=logging.DEBUG)

    if len(args.data_root) == 0:
        logging.error("-data_root is required to run downloader!")
        exit()

    if not os.path.isdir(args.data_root):
        logging.error(f'folder {args.data_root} does not exist! please provide existing folder in -data_root arg!')
        exit()


    IMAGENET_API_WNID_TO_URLS = lambda wnid: f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={wnid}'

    current_folder = os.path.dirname(os.path.realpath(__file__))

    class_info_json_filename = 'imagenet_class_info.json'
    class_info_json_filepath = os.path.join(current_folder, class_info_json_filename)

    class_info_dict = dict()

    with open(class_info_json_filepath) as class_info_json_f:
        class_info_dict = json.load(class_info_json_f)

    classes_to_scrape = []

    if args.use_class_list == True:
       for item in args.class_list:
           classes_to_scrape.append(item)
           if item not in class_info_dict:
               logging.error(f'Class {item} not found in ImageNete')
               exit()

    elif args.use_class_list == False:
        potential_class_pool = []
        for key, val in class_info_dict.items():

            if args.scrape_only_flickr:
                if int(val['flickr_img_url_count']) * 0.9 > args.images_per_class:
                    potential_class_pool.append(key)
            else:
                if int(val['img_url_count']) * 0.8 > args.images_per_class:
                    potential_class_pool.append(key)

        if (len(potential_class_pool) < args.number_of_classes):
            logging.error(f"With {args.images_per_class} images per class there are {len(potential_class_pool)} to choose from.")
            logging.error(f"Decrease number of classes or decrease images per class.")
            exit()

        picked_classes_idxes = np.random.choice(len(potential_class_pool), args.number_of_classes, replace = False)

        for idx in picked_classes_idxes:
            classes_to_scrape.append(potential_class_pool[idx])


    print("Picked the following clases:")
    print([ class_info_dict[class_wnid]['class_name'] for class_wnid in classes_to_scrape ])


    imagenet_images_folder = os.path.join(args.data_root, 'imagenet_images')
    if not os.path.isdir(imagenet_images_folder):
        os.mkdir(imagenet_images_folder)



    scraping_stats = dict(
        all=dict(
            tried=0,
            success=0,
            time_spent=0,
        ),
        is_flickr=dict(
            tried=0,
            success=0,
            time_spent=0,
        ),
        not_flickr=dict(
            tried=0,
            success=0,
            time_spent=0,
        )
    )

    def add_debug_csv_row(row):
        with open('stats.csv', "a") as csv_f:
            csv_writer = csv.writer(csv_f, delimiter=",")
            csv_writer.writerow(row)

    class MultiStats():
        def __init__(self):

            self.lock = Lock()

            self.stats = dict(
                all=dict(
                    tried=Value('d', 0),
                    success=Value('d',0),
                    time_spent=Value('d',0),
                ),
                is_flickr=dict(
                    tried=Value('d', 0),
                    success=Value('d',0),
                    time_spent=Value('d',0),
                ),
                not_flickr=dict(
                    tried=Value('d', 0),
                    success=Value('d', 0),
                    time_spent=Value('d', 0),
                )
            )
        def inc(self, cls, stat, val):
            with self.lock:
                self.stats[cls][stat].value += val

        def get(self, cls, stat):
            with self.lock:
                ret = self.stats[cls][stat].value
            return ret

    multi_stats = MultiStats()


    if args.debug:
        row = [
            "all_tried",
            "all_success",
            "all_time_spent",
            "is_flickr_tried",
            "is_flickr_success",
            "is_flickr_time_spent",
            "not_flickr_tried",
            "not_flickr_success",
            "not_flickr_time_spent"
        ]
        add_debug_csv_row(row)

    def add_stats_to_debug_csv():
        row = [
            multi_stats.get('all', 'tried'),
            multi_stats.get('all', 'success'),
            multi_stats.get('all', 'time_spent'),
            multi_stats.get('is_flickr', 'tried'),
            multi_stats.get('is_flickr', 'success'),
            multi_stats.get('is_flickr', 'time_spent'),
            multi_stats.get('not_flickr', 'tried'),
            multi_stats.get('not_flickr', 'success'),
            multi_stats.get('not_flickr', 'time_spent'),
        ]
        add_debug_csv_row(row)

    def print_stats(cls, print_func):

        actual_all_time_spent = time.time() - scraping_t_start.value
        processes_all_time_spent = multi_stats.get('all', 'time_spent')

        if processes_all_time_spent == 0:
            actual_processes_ratio = 1.0
        else:
            actual_processes_ratio = actual_all_time_spent / processes_all_time_spent

        #print(f"actual all time: {actual_all_time_spent} proc all time {processes_all_time_spent}")

        print_func(f'STATS For class {cls}:')
        print_func(f' tried {multi_stats.get(cls, "tried")} urls with'
                   f' {multi_stats.get(cls, "success")} successes')

        if multi_stats.get(cls, "tried") > 0:
            print_func(f'{100.0 * multi_stats.get(cls, "success")/multi_stats.get(cls, "tried")}% success rate for {cls} urls ')
        if multi_stats.get(cls, "success") > 0:
            print_func(f'{multi_stats.get(cls,"time_spent") * actual_processes_ratio / multi_stats.get(cls,"success")} seconds spent per {cls} succesful image download')



    lock = Lock()
    url_tries = Value('d', 0)
    scraping_t_start = Value('d', time.time())
    class_folder = ''
    class_images = Value('d', 0)

    def get_image(img_url):

        #print(f'Processing {img_url}')

        #time.sleep(3)

        if len(img_url) <= 1:
            return


        cls_imgs = 0
        with lock:
            cls_imgs = class_images.value

        if cls_imgs >= args.images_per_class:
            return

        logging.debug(img_url)

        cls = ''

        if 'flickr' in img_url:
            cls = 'is_flickr'
        else:
            cls = 'not_flickr'
            if args.scrape_only_flickr:
                return

        t_start = time.time()

        def finish(status):
            t_spent = time.time() - t_start
            multi_stats.inc(cls, 'time_spent', t_spent)
            multi_stats.inc('all', 'time_spent', t_spent)

            multi_stats.inc(cls,'tried', 1)
            multi_stats.inc('all', 'tried', 1)

            if status == 'success':
                multi_stats.inc(cls,'success', 1)
                multi_stats.inc('all', 'success', 1)

            elif status == 'failure':
                pass
            else:
                logging.error(f'No such status {status}!!')
                exit()
            return


        with lock:
            url_tries.value += 1
            if url_tries.value % 250 == 0:
                print(f'\nScraping stats:')
                print_stats('is_flickr', print)
                print_stats('not_flickr', print)
                print_stats('all', print)
                if args.debug:
                    add_stats_to_debug_csv()

        try:
            img_resp = requests.get(img_url, timeout = 1)
        except ConnectionError:
            logging.debug(f"Connection Error for url {img_url}")
            return finish('failure')
        except ReadTimeout:
            logging.debug(f"Read Timeout for url {img_url}")
            return finish('failure')
        except TooManyRedirects:
            logging.debug(f"Too many redirects {img_url}")
            return finish('failure')
        except MissingSchema:
            return finish('failure')
        except InvalidURL:
            return finish('failure')

        if not 'content-type' in img_resp.headers:
            return finish('failure')

        if not 'image' in img_resp.headers['content-type']:
            logging.debug("Not an image")
            return finish('failure')

        if (len(img_resp.content) < 1000):
            return finish('failure')

        logging.debug(img_resp.headers['content-type'])
        logging.debug(f'image size {len(img_resp.content)}')

        img_name = img_url.split('/')[-1]
        img_name = img_name.split("?")[0]

        if (len(img_name) <= 1):
            return finish('failure')

        img_file_path = os.path.join(class_folder, img_name)
        logging.debug(f'Saving image in {img_file_path}')

        with open(img_file_path, 'wb') as img_f:
            img_f.write(img_resp.content)

            with lock:
                class_images.value += 1

            logging.debug(f'Scraping stats')
            print_stats('is_flickr', logging.debug)
            print_stats('not_flickr', logging.debug)
            print_stats('all', logging.debug)

            return finish('success')


    for class_wnid in classes_to_scrape:

        class_name = class_info_dict[class_wnid]["class_name"]
        print(f'Scraping images for class \"{class_name}\"')
        url_urls = IMAGENET_API_WNID_TO_URLS(class_wnid)

        time.sleep(0.75)
        resp = requests.get(url_urls)

        class_folder = os.path.join(imagenet_images_folder, class_name)
        if not os.path.exists(class_folder):
            os.mkdir(class_folder)

        class_images.value = 0

        urls = [url.decode('utf-8') for url in resp.content.splitlines()]

        #for url in  urls:
        #    get_image(url)

        print(f"Multiprocessing workers: {args.multiprocessing_workers}")
        with Pool(processes=args.multiprocessing_workers) as p:
            p.map(get_image,urls)

        logging.info("[downloaded]%s:%i"%(class_name,int(multi_stats.get("all","success"))))



if __name__ == '__main__':

    # Unbalanced Imagenet 2012 classification classes
    class_list="n00006484 n00471613 n00476389 n00480211 n00480993 n01318381 n01321123 n01323493 n01440764 n01442450 n01443537 n01484850 n01486010 n01491361 n01494475 n01496331 n01498041 n01514668 n01514859 n01518878 n01530575 n01531178 n01531344 n01532829 n01534433 n01537544 n01558993 n01560419 n01562265 n01580077 n01582220 n01587834 n01592084 n01601694 n01608432 n01614925 n01616318 n01622779 n01627424 n01629819 n01629962 n01630670 n01631663 n01632458 n01632777 n01634522 n01641577 n01644373 n01644900 n01664065 n01665541 n01667114 n01667778 n01669191 n01675722 n01677366 n01682714 n01685808 n01687978 n01688243 n01689811 n01692333 n01693334 n01694178 n01694709 n01695060 n01697457 n01698640 n01704323 n01728572 n01728920 n01729322 n01729977 n01730563 n01730812 n01734418 n01735189 n01737021 n01737875 n01739381 n01740131 n01741943 n01742172 n01744401 n01748264 n01749939 n01751748 n01753032 n01753488 n01755581 n01756291 n01768244 n01770081 n01770393 n01773157 n01773549 n01773797 n01774384 n01774750 n01775062 n01776313 n01784675 n01792158 n01792640 n01795545 n01796340 n01797886 n01798484 n01806143 n01806567 n01807496 n01817953 n01818515 n01819313 n01820546 n01824575 n01828970 n01829413 n01833805 n01843065 n01843383 n01847000 n01855032 n01855672 n01860187 n01871265 n01872401 n01872772 n01873310 n01877812 n01882714 n01883070 n01889520 n01893825 n01910747 n01914609 n01917289 n01924916 n01930112 n01943899 n01944390 n01945685 n01950731 n01955084 n01968897 n01978287 n01978455 n01980166 n01981276 n01983481 n01984695 n01985128 n01986214 n01990800 n02002556 n02002724 n02006656 n02007558 n02008643 n02009229 n02009750 n02009912 n02011460 n02012849 n02013706 n02017213 n02018207 n02018795 n02025239 n02027492 n02028035 n02028900 n02033041 n02037110 n02051845 n02056570 n02058221 n02066245 n02071028 n02071294 n02074367 n02077923 n02085620 n02085782 n02085936 n02086079 n02086240 n02086646 n02086910 n02087046 n02087394 n02088094 n02088238 n02088364 n02088466 n02088632 n02089078 n02089867 n02089973 n02090379 n02090622 n02090721 n02091032 n02091134 n02091244 n02091467 n02091635 n02091831 n02092002 n02092339 n02093256 n02093428 n02093647 n02093754 n02093859 n02093991 n02094114 n02094258 n02094433 n02095314 n02095570 n02095889 n02096051 n02096177 n02096294 n02096437 n02096585 n02097047 n02097130 n02097209 n02097298 n02097474 n02097658 n02098105 n02098286 n02098413 n02099267 n02099429 n02099601 n02099712 n02099849 n02100236 n02100583 n02100735 n02100877 n02101006 n02101388 n02101556 n02102040 n02102177 n02102318 n02102480 n02102973 n02104029 n02104365 n02105056 n02105162 n02105251 n02105412 n02105505 n02105641 n02105855 n02106030 n02106166 n02106382 n02106550 n02106662 n02107142 n02107312 n02107574 n02107683 n02107908 n02108000 n02108089 n02108422 n02108551 n02108915 n02109047 n02109525 n02109961 n02110063 n02110185 n02110341 n02110627 n02110806 n02110958 n02111129 n02111277 n02111500 n02111889 n02112018 n02112137 n02112350 n02112706 n02113023 n02113186 n02113624 n02113712 n02113799 n02113978 n02114367 n02114548 n02114712 n02114855 n02115641 n02115913 n02116738 n02117135 n02119022 n02119477 n02119634 n02119789 n02120079 n02120505 n02122878 n02123045 n02123159 n02123394 n02123478 n02123597 n02124075 n02124157 n02125311 n02127052 n02128385 n02128669 n02128757 n02128925 n02129165 n02129604 n02130308 n02132136 n02132320 n02133161 n02134084 n02134418 n02137549 n02138441 n02165105 n02165456 n02167151 n02168699 n02169497 n02172182 n02174001 n02177972 n02190166 n02206856 n02219486 n02226429 n02229544 n02231487 n02233338 n02236044 n02256656 n02259212 n02264363 n02268853 n02276258 n02277742 n02279257 n02279972 n02280649 n02281406 n02281787 n02317335 n02319095 n02321529 n02325366 n02326432 n02328150 n02330245 n02342885 n02346627 n02356798 n02361337 n02363005 n02364673 n02378969 n02382132 n02382204 n02384858 n02389026 n02389943 n02391049 n02395406 n02396014 n02396427 n02397096 n02398521 n02402175 n02403003 n02408429 n02410509 n02412080 n02412787 n02415577 n02417387 n02417914 n02422106 n02422699 n02423022 n02437312 n02437616 n02441942 n02442845 n02443114 n02443346 n02443484 n02444819 n02445715 n02447366 n02454379 n02457408 n02460009 n02480495 n02480855 n02481823 n02483362 n02483708 n02484975 n02486261 n02486410 n02487347 n02488291 n02488702 n02489166 n02490219 n02492035 n02492660 n02493509 n02493793 n02494079 n02497673 n02500267 n02504013 n02504458 n02509815 n02510455 n02514041 n02526121 n02536864 n02576575 n02606052 n02607072 n02630739 n02640242 n02641379 n02643566 n02655020 n02666196 n02667093 n02672831 n02676566 n02687172 n02690373 n02692877 n02699494 n02699629 n02701002 n02704645 n02704792 n02708093 n02727426 n02730930 n02747177 n02749292 n02749479 n02769748 n02776631 n02777292 n02782093 n02783161 n02786058 n02787622 n02788148 n02790996 n02791124 n02791270 n02793495 n02794156 n02795169 n02795528 n02797295 n02799071 n02802426 n02804414 n02804515 n02804610 n02806530 n02807133 n02808304 n02808440 n02814533 n02814860 n02815749 n02815834 n02815950 n02817516 n02818135 n02818254 n02823428 n02823750 n02824448 n02825657 n02834397 n02835271 n02837789 n02840134 n02840245 n02841315 n02843684 n02855089 n02859443 n02860847 n02861022 n02864987 n02865351 n02869837 n02870880 n02871525 n02877765 n02879517 n02879718 n02880189 n02883205 n02891788 n02892201 n02892304 n02892767 n02894605 n02895154 n02906734 n02909053 n02909870 n02910353 n02916350 n02916936 n02917067 n02918330 n02927161 n02930766 n02931013 n02931148 n02939185 n02948072 n02950256 n02950482 n02950632 n02950826 n02951358 n02951585 n02963159 n02965783 n02966193 n02968473 n02969010 n02969886 n02971356 n02974003 n02977058 n02978881 n02979186 n02980441 n02980625 n02981792 n02988304 n02991048 n02991302 n02992211 n02992529 n02999410 n02999936 n03000134 n03000247 n03000684 n03014705 n03016953 n03017168 n03018349 n03026506 n03028079 n03032252 n03041632 n03042490 n03045337 n03045698 n03047690 n03049457 n03055670 n03062245 n03063599 n03063689 n03065424 n03065708 n03066232 n03075370 n03075768 n03085013 n03089624 n03095699 n03100240 n03100346 n03109150 n03110669 n03114504 n03124043 n03124170 n03125729 n03126707 n03127747 n03127925 n03131574 n03131669 n03132666 n03132776 n03133878 n03134739 n03141823 n03146219 n03147509 n03160309 n03162556 n03175189 n03179701 n03180011 n03187595 n03188531 n03188725 n03196217 n03197337 n03201208 n03204306 n03207743 n03207941 n03208938 n03216828 n03218198 n03220513 n03220692 n03223299 n03240683 n03249569 n03249956 n03250847 n03255030 n03258456 n03258577 n03259280 n03259401 n03271574 n03272010 n03272562 n03290653 n03291741 n03291819 n03291963 n03297495 n03314780 n03325584 n03336839 n03337140 n03344393 n03345487 n03347037 n03355925 n03372029 n03372549 n03373237 n03376595 n03379051 n03384352 n03388043 n03388183 n03388549 n03393912 n03394916 n03400231 n03404251 n03417042 n03424325 n03425413 n03443371 n03444034 n03445777 n03445924 n03447447 n03447721 n03450230 n03450516 n03450734 n03452741 n03457902 n03459775 n03461385 n03467068 n03476684 n03476991 n03478589 n03481172 n03481521 n03482001 n03482128 n03482405 n03483316 n03485407 n03485794 n03492542 n03494278 n03495258 n03495570 n03496892 n03498962 n03501288 n03501520 n03501614 n03527444 n03527565 n03529860 n03530642 n03532342 n03532672 n03532919 n03534580 n03535284 n03535780 n03536761 n03537085 n03537241 n03538406 n03544143 n03584254 n03584829 n03585073 n03585337 n03588951 n03589313 n03589513 n03589672 n03594734 n03594945 n03595523 n03595614 n03598930 n03599486 n03602883 n03617480 n03623198 n03627232 n03630383 n03633091 n03637318 n03642806 n03649909 n03657121 n03658185 n03660909 n03661043 n03662601 n03665366 n03666591 n03670208 n03673027 n03673270 n03676483 n03680355 n03690938 n03691459 n03697007 n03706229 n03709644 n03709823 n03709960 n03710193 n03710637 n03710721 n03717622 n03720891 n03721384 n03724870 n03725035 n03729826 n03733131 n03733281 n03733805 n03742115 n03743016 n03759954 n03761084 n03763968 n03764736 n03769881 n03770439 n03770679 n03770954 n03773504 n03775071 n03775546 n03776460 n03777568 n03777754 n03781244 n03781787 n03782006 n03782190 n03785016 n03786715 n03786901 n03787032 n03788195 n03788365 n03791053 n03792782 n03792972 n03793489 n03794056 n03796401 n03803284 n03804744 n03811295 n03814639 n03814906 n03825788 n03832673 n03836062 n03837869 n03838899 n03840681 n03841143 n03843555 n03854065 n03857828 n03866082 n03868242 n03868863 n03871628 n03873416 n03873699 n03873848 n03874293 n03874599 n03876231 n03877845 n03878066 n03878211 n03884397 n03887697 n03888257 n03888605 n03891251 n03891332 n03895866 n03899768 n03902125 n03903868 n03908618 n03908714 n03916031 n03920288 n03924679 n03929202 n03929443 n03929660 n03929855 n03930313 n03930630 n03933933 n03934042 n03935335 n03937543 n03938244 n03942813 n03944138 n03944341 n03947888 n03950228 n03955809 n03955941 n03956157 n03958227 n03959701 n03960374 n03960490 n03961711 n03967562 n03976467 n03976657 n03977158 n03977966 n03980874 n03982430 n03983396 n03990474 n03991062 n03995372 n03998194 n03999992 n04000311 n04000480 n04004475 n04004767 n04005630 n04008634 n04009552 n04009801 n04019541 n04023962 n04026417 n04033901 n04033995 n04037443 n04039381 n04040759 n04041069 n04041544 n04044716 n04049303 n04065272 n04067472 n04067658 n04069434 n04070727 n04074963 n04080833 n04081281 n04086273 n04090263 n04091097 n04093157 n04098513 n04098795 n04099969 n04111414 n04111531 n04116512 n04118538 n04118776 n04120489 n04122578 n04125021 n04125116 n04127249 n04131690 n04133789 n04136333 n04140777 n04141076 n04141327 n04141838 n04141975 n04146614 n04147183 n04147291 n04149374 n04149813 n04151581 n04151940 n04152387 n04152593 n04153751 n04154152 n04154340 n04154565 n04162706 n04179913 n04180063 n04192698 n04192858 n04200800 n04201297 n04204238 n04204347 n04208210 n04208427 n04209133 n04209239 n04215910 n04228054 n04229816 n04235860 n04238763 n04239074 n04243546 n04250850 n04251144 n04252077 n04252225 n04254120 n04254680 n04254777 n04258138 n04259630 n04263257 n04264628 n04265275 n04266014 n04269944 n04270147 n04273569 n04275661 n04275904 n04277352 n04277493 n04277669 n04285008 n04286128 n04286575 n04296562 n04310018 n04311004 n04311174 n04317175 n04325704 n04326547 n04328186 n04330267 n04330340 n04332243 n04335435 n04336792 n04337157 n04344873 n04346328 n04347754 n04350905 n04355338 n04355933 n04356056 n04357314 n04366367 n04367480 n04367746 n04370456 n04371430 n04371774 n04372370 n04376876 n04380533 n04388743 n04389033 n04392985 n04398044 n04404412 n04409515 n04417672 n04418357 n04423845 n04428191 n04429376 n04435653 n04442312 n04443257 n04447861 n04456115 n04456472 n04456734 n04457157 n04458633 n04461696 n04462240 n04465501 n04465666 n04467665 n04476259 n04479046 n04482393 n04483307 n04485082 n04486054 n04487081 n04487394 n04493381 n04501370 n04504141 n04505470 n04507155 n04509417 n04515003 n04517823 n04522168 n04523525 n04523831 n04524716 n04525038 n04525305 n04532106 n04532670 n04536866 n04540053 n04542943 n04543158 n04543509 n04546194 n04548280 n04548362 n04550184 n04552348 n04553703 n04554684 n04554871 n04557648 n04560804 n04562935 n04579145 n04579432 n04579667 n04584207 n04589890 n04590129 n04591157 n04591713 n04592741 n04596742 n04597913 n04599235 n04604644 n04606251 n04612373 n04612504 n04613696 n04951373 n04965179 n05261310 n05538625 n05716342 n06263369 n06275353 n06277135 n06277280 n06359193 n06596364 n06785654 n06794110 n06874185 n07248320 n07273802 n07565083 n07565161 n07565259 n07579787 n07583066 n07584110 n07590611 n07596967 n07607605 n07613480 n07614500 n07615774 n07684084 n07693725 n07695742 n07697313 n07697537 n07711569 n07712063 n07714571 n07714990 n07715103 n07716358 n07716906 n07717410 n07717556 n07718472 n07718747 n07720875 n07730033 n07731952 n07734744 n07736371 n07742313 n07745940 n07747607 n07749582 n07753113 n07753275 n07753592 n07754684 n07760859 n07768694 n07802026 n07831146 n07836838 n07838551 n07860988 n07871810 n07873807 n07875152 n07880968 n07892512 n07915213 n07917272 n07920052 n07922607 n07930864 n07932039 n07977870 n08376250 n08496334 n09193705 n09218315 n09229709 n09246464 n09248153 n09256479 n09283767 n09288635 n09332890 n09399592 n09403211 n09421951 n09428293 n09456207 n09468604 n09472597 n09689435 n09835506 n09931418 n09931640 n10147935 n10148035 n10149436 n10154601 n10393909 n10435988 n10492727 n10565667 n10630188 n10768903 n10792856 n11693981 n11725015 n11875938 n11876634 n11876803 n11879895 n11939491 n11959632 n11959862 n12143676 n12144580 n12159804 n12160303 n12160857 n12161744 n12165384 n12267677 n12345280 n12352287 n12400720 n12401684 n12607456 n12620546 n12708293 n12711596 n12751172 n12768369 n12768682 n12786836 n12985857 n12997919 n12998815 n13000891 n13001041 n13037406 n13040303 n13044778 n13052670 n13054073 n13054560 n13133613 n13354021 n13869547 n13869788 n13876561 n13886260 n13896100 n13901211 n13902793 n14765422 n15075141"
    c=class_list.split()
    c.reverse()
    class_list=" ".join(c)
    parser = argparse.ArgumentParser(description='ImageNet image scraper')
    parser.add_argument('-scrape_only_flickr', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-number_of_classes', default = 100, type=int)
    parser.add_argument('-images_per_class', default = 1000, type=int)
    parser.add_argument('-data_root', default='dataset', type=str)
    parser.add_argument('-use_class_list', default=False,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-class_list', default=class_list.split(), nargs='*')
    parser.add_argument('-debug', default=False,type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('-multiprocessing_workers', default = 24, type=int)

    args, args_other = parser.parse_known_args()
    
    main(args)
