package com.example.demo;

import com.example.demo.Entity.tweet;
import com.example.demo.Entity.tweetDto;
import com.example.demo.Entity.tweetETO;
import com.example.demo.Mapper.tweetDtoMapper;
import com.example.demo.Mapper.tweetETOMapper;
import com.example.demo.Mapper.tweetMapper;
import com.example.demo.Mapper.weiboMapper;
import org.junit.jupiter.api.Test;
import org.omg.CORBA.PRIVATE_MEMBER;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestExecutionListeners;

import java.rmi.MarshalledObject;
import java.text.SimpleDateFormat;
import java.time.ZonedDateTime;
import java.util.*;

@SpringBootTest
class Demo1ApplicationTests {
    @Autowired
    private weiboMapper weiboMapper;

    @Autowired
    private tweetMapper tweetMapper;

    @Autowired
    private tweetDtoMapper tweetDtoMapper;

    @Autowired
    private tweetETOMapper tweetETOMapper;


    @Test
    public void Testval() {
        System.out.println(weiboMapper.SqlString("select count(*) from weibo;"));

    }

    @Test
    public void testmap() {
        ArrayList<String> arr = new ArrayList<>();

        Map map = new HashMap();


        String[] location = new String[]{"青海", "西藏", "上海", "云南", "内蒙古", "其他", "北京", "台湾", "吉林", "四川", "天津", "宁夏", "安徽", "山东", "山西", "广东", "广西", "新疆", "江苏", "江西", "河北", "河南", "浙江", "海南", "海外", "湖北", "湖南", "澳门", "甘肃", "福建", "贵州", "陕西", "辽宁", "重庆", "香港", "黑龙江"};
        for (String area : location) {
            map.put(area, Integer.parseInt(weiboMapper.SqlString("select count(location) from weibo where location like  '%" + area + "%';")));
        }

        List<Map.Entry<String, Integer>> infoIds = new ArrayList<Map.Entry<String, Integer>>(map.entrySet());
        Collections.sort(infoIds, new Comparator<Map.Entry<String, Integer>>() {
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return (o2.getValue() - o1.getValue());
                //return (o1.getKey()).toString().compareTo(o2.getKey());
            }
        });
//排序后
        for (int i = 0; i < infoIds.size(); i++) {
            String id = infoIds.get(i).toString();
            System.out.println(id);
            String value = infoIds.get(i).getKey();


        }

    }
    @Test
    void testcloud(){
        System.out.println(weiboMapper.FindText("select word,overview  from frequency order by overview desc limit 0,20;"));
    }



}
