package com.example.demo.Controller;

import com.example.demo.Service.tweetService;
import com.example.demo.Service.weiboService;
import com.example.demo.common.returnvalue;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/weibo")
public class weibocontroller {
    @Autowired
    private weiboService weiboervice;


    @CrossOrigin
    @GetMapping
    public returnvalue<Map> findText(){
        Map<Object,Object> map=new HashMap<>();
        Map<Object,Object> WeiboPositiveText=new HashMap<>();
        Map<Object,Object> WeiboNegativeText=new HashMap<>();
        Map<Object,Object> WeiboText=new HashMap<>();

        WeiboText.put("WeiboText",weiboervice.FindText());
        WeiboPositiveText.put("TweetPositiveText",weiboervice.FindPositiveText());
        WeiboNegativeText.put("TweetNegativeText",weiboervice.FindNegativeText());
        map.put("TweetPositiveText",WeiboPositiveText);
        map.put("TweetNegativeText",WeiboNegativeText);
        map.put("WeiboText",WeiboText);
        map.put("WeiboCloud",weiboervice.FindOverviewCloud());
        Map LocationMap=new HashMap<>();
        LocationMap=weiboervice.findmap();
        map.put("locationMap",LocationMap);
        return returnvalue.success(map);
    }
}
