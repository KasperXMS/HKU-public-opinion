package com.example.demo.Controller;

import com.example.demo.Service.tweetService;
import com.example.demo.Service.weiboService;
import com.example.demo.common.returnvalue;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.support.incrementer.HsqlMaxValueIncrementer;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/overview")
public class overviewcontroller {
    @Autowired
    private tweetService tweetService;

    @Autowired
    private weiboService weiboervice;

    //bar chart of weibo and tweet
    @CrossOrigin
    @GetMapping
    public returnvalue<Map> barchart(){
        Map<Object,Object> map=new HashMap<>();

        //data of the barchart
        int tweetCount=Integer.parseInt(tweetService.countnumber());
        int weiboCount=Integer.parseInt(weiboervice.countnumber());
        Map<Object,Object> barchart=new HashMap<>();
        barchart.put("TweetCount",tweetCount);
        barchart.put("WeiboCount",weiboCount);
        map.put("barchart",barchart);

        //data of the linechart
        String tweetAvg=tweetService.AvgValue();
        String weiboAvg=weiboervice.AvgValue();
        Map<Object,Object> linechart=new HashMap<>();
        linechart.put("TweetPositiveVal",tweetAvg);
        linechart.put("WeiboPositiveVal",weiboAvg);
        map.put("linechart",linechart);



        Map<Object,Object> DateVal=new HashMap<>();
        Map<Object,Object> TweetDateVal=new HashMap<>();
        Map<Object,Object> WeiboDateVal=new HashMap<>();
        TweetDateVal=tweetService.DateVal();
        WeiboDateVal=weiboervice.DateVal();
        DateVal.put("TweetDateVal",TweetDateVal);
        DateVal.put("WeiboDateVal",WeiboDateVal);
        map.put("DateVal",DateVal);

        Map<Object,Object> HotVal=new HashMap<>();
        Map<Object,Object> TweetHotVal=new HashMap<>();
        Map<Object,Object> WeiboHotVal=new HashMap<>();
        TweetHotVal=tweetService.Hotval();
        WeiboHotVal=weiboervice.HotVal();
        HotVal.put("TweetHotVal",TweetHotVal);
        HotVal.put("WeiboHotVal",WeiboHotVal);
        map.put("HotVal",HotVal);

        Map<Object,Object> TweetPositiveText=new HashMap<>();
        Map<Object,Object> TweetNegativeText=new HashMap<>();
        Map<Object,Object> WeiboPositiveText=new HashMap<>();
        Map<Object,Object> WeiboNegativeText=new HashMap<>();
        TweetPositiveText.put("TweetPositiveText",tweetService.FindPositiveText());
        TweetNegativeText.put("TweetNegativeText",tweetService.FindNegativeText());
        WeiboPositiveText.put("WeiboPositiveText",tweetService.FindPositiveText());
        WeiboNegativeText.put("WeiboNegativeText",tweetService.FindNegativeText());


        map.put("OverviewCloud",weiboervice.FindOverviewCloud());



        return returnvalue.success(map);
    }
}
