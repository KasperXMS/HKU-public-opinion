package com.example.demo.Service.Impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.demo.Entity.tweet;
import com.example.demo.Entity.tweetDto;
import com.example.demo.Entity.tweetETO;
import com.example.demo.Entity.wordcloud;
import com.example.demo.Mapper.tweetDtoMapper;
import com.example.demo.Mapper.tweetETOMapper;
import com.example.demo.Mapper.tweetMapper;

import com.example.demo.Mapper.wordclouMapper;
import com.example.demo.Service.tweetService;
import javafx.beans.binding.ObjectExpression;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class tweetServiceImpl extends ServiceImpl<tweetMapper,tweet> implements tweetService {
    @Autowired
    private tweetService tweetService;

    @Autowired
    private tweetMapper tweetMapper;

    @Autowired
    private tweetDtoMapper tweetDtoMapper;

    @Autowired
    private tweetETOMapper tweetETOMapper;

    @Autowired
    private wordclouMapper wordcloudMapper;


    @Override
    public String countnumber() {
        return tweetMapper.SqlString("select count(*) from tweet_hku;");
    }

    @Override
    public String AvgValue() {

        return tweetMapper.SqlString("select AVG(0.6*positive+0.4*neutral) from tweet_hku where keyword is not null;");
    }

    @Override
    public Map DateVal() {
        Map map = new HashMap();
        Calendar cal = Calendar.getInstance();
        for(int i=0;i<7;i++) {

            cal.add(Calendar.DATE, -1);
            String time = new SimpleDateFormat("yyyy-MM-dd ").format(cal.getTime());
            int m=i+1;
            map.put(time, Integer.parseInt(tweetMapper.SqlString("SELECT count(*) FROM tweet_hku WHERE DATEDIFF(time,NOW())=-" + m)));

        }
        return map;
    }

    @Override
    public Map Hotval() {
        Map map = new HashMap();
        Calendar cal = Calendar.getInstance();
        for(int i=0;i<7;i++) {

            cal.add(Calendar.DATE, -1);
            String time = new SimpleDateFormat("yyyy-MM-dd ").format(cal.getTime());
            int m=i+1;
            map.put(time, tweetMapper.SqlString("SELECT sum(favorite_count) FROM tweet_hku WHERE DATEDIFF(time,NOW())=-" + m));

        }
        return map;
    }

    @Override
    public List FindPositiveText() {
        List<tweetDto> list=tweetDtoMapper.FindText("select distinct text,favorite_count ,(positive*0.6 + 0.4*neutral) as score from tweet_hku  where (positive*0.6+tweet_hku.neutral*0.4)>0.35 order by favorite_count desc limit  0,20;");
        return  list;

     }

    @Override
    public List FindNegativeText() {
        List<tweetDto> list=tweetDtoMapper.FindText("select distinct text,favorite_count ,(tweet_hku.negative*0.6 + 0.4*neutral) as score from tweet_hku  where (tweet_hku.negative*0.6+tweet_hku.neutral*0.4)>0.35 order by favorite_count desc limit  0,20;");
        return  list;
    }

    @Override
    public List FindText() {
        List<tweetETO> list = tweetETOMapper.FindText("select text,favorite_count,(tweet_hku.positive*0.6 + 0.4*neutral) as score from tweet_hku order by favorite_count desc limit  0,20;");
        return list;
    }

    @Override
    public List FindWordCloud() {
        List<wordcloud> list =wordcloudMapper.FindWordCloud("select word,overview  from frequency where source='tweet_hku' order by overview desc limit 0,20;");
        return list;
    }


}
