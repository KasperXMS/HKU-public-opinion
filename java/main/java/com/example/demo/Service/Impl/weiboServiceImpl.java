package com.example.demo.Service.Impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.demo.Entity.*;
import com.example.demo.Mapper.weiboDtoMapper;
import com.example.demo.Mapper.weiboETOMapper;
import com.example.demo.Mapper.weiboMapper;
import com.example.demo.Mapper.wordclouMapper;
import com.example.demo.Service.weiboService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.text.SimpleDateFormat;
import java.util.*;

@Service
@Slf4j
public class weiboServiceImpl extends ServiceImpl<weiboMapper, weibo> implements weiboService {


    @Autowired
    private weiboService weiboService;
    @Autowired
    private weiboMapper weiboMapper;
    @Autowired
    private weiboDtoMapper weiboDtoMapper;
    @Autowired
    private weiboETOMapper weiboETOMapper;
    @Autowired
    private wordclouMapper wordcloudMapper;

    @Override
    public String countnumber() {
        return weiboMapper.SqlString("select count(*) from weibo;");
    }

    @Override
    public String AvgValue() {
        return weiboMapper.SqlString("select AVG(positive) from weibo where keyword is not null;");
    }

    @Override
    public Map findmap() {

        Map map = new HashMap();
        String[] location=new String[]{"青海","西藏","上海","云南","内蒙古","其他","北京","台湾","吉林","四川",
                "天津","宁夏","安徽","山东","山西","广东","广西","新疆","江苏","江西","河北","河南","浙江","海南","海外","湖北","湖南","澳门","甘肃","福建","贵州","陕西","辽宁","重庆","香港","黑龙江"};

        for(String area : location){
            map.put(area, Integer.parseInt(weiboMapper.SqlString("select count(location) from weibo where location like  '%"+area+"%';")));
        }


        List<Map.Entry<String, Integer>> infoIds = new ArrayList<Map.Entry<String, Integer>>(map.entrySet());
        Collections.sort(infoIds, new Comparator<Map.Entry<String, Integer>>() {
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return (o2.getValue() - o1.getValue());
                //return (o1.getKey()).toString().compareTo(o2.getKey());
            }
        });
        HashMap<Object,Integer> hashMap =new HashMap<>();
        for (int i = 0; i < infoIds.size(); i++) {
            String area = infoIds.get(i).getKey();
            Integer value= (Integer) map.get(area);
            hashMap.put(area,value);

        }



        return hashMap;
    }

    @Override
    public Map DateVal() {
       // ZonedDateTime ZonedDateTime = ZonedDateTime.now();
        Map map = new HashMap();
        Calendar cal = Calendar.getInstance();
        for(int i=0;i<7;i++) {

            cal.add(Calendar.DATE, -1);
            String time = new SimpleDateFormat("yyyy-MM-dd ").format(cal.getTime());
            int m=i+1;
            map.put(time, Integer.parseInt(weiboMapper.SqlString("SELECT count(*) FROM weibo WHERE DATEDIFF(time,NOW())=-" + m)));

        }
        return map;
    }

    @Override
    public Map HotVal() {
        Map map = new HashMap();
        Calendar cal = Calendar.getInstance();
        for(int i=0;i<7;i++) {

            cal.add(Calendar.DATE, -1);
            String time = new SimpleDateFormat("yyyy-MM-dd ").format(cal.getTime());
            int m=i+1;
            map.put(time, Integer.parseInt(weiboMapper.SqlString("SELECT sum(likes) FROM weibo WHERE DATEDIFF(time,NOW())=-" + m)));

        }
        return map;
    }

    @Override
    public List FindPositiveText() {
        List<weiboDto> list= new ArrayList<>();
        list=weiboDtoMapper.FindText("select text, likes ,positive as score from weibo  where positive>0.85  order by likes desc limit  0,20;");
        return list;
    }

    @Override
    public List FindNegativeText() {
        List<weiboDto> list= new ArrayList<>();
        list=weiboDtoMapper.FindText("select text, likes ,negative as score from weibo  where negative>0.85  order by likes desc limit  0,20;");
        return list;
    }

    @Override
    public List FindText() {
        List<weiboETO> list = weiboETOMapper.FindText("select text,likes,positive as score from weibo order by likes desc limit  0,20;");
        return list;
    }

    @Override
    public List FindWordCloud() {
        List<wordcloud> list =wordcloudMapper.FindWordCloud("select word,overview  from frequency where source='weibo' order by overview desc limit 0,20;");
        return list;
    }

    @Override
    public List FindOverviewCloud() {
        List<wordcloud> list =wordcloudMapper.FindWordCloud("select word,overview  from frequency order by overview desc limit 0,20;");
        return list;
    }


}
