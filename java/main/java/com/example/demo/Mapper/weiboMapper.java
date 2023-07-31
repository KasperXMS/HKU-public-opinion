package com.example.demo.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.Entity.weibo;
import com.example.demo.Entity.weiboETO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

@Mapper
public interface weiboMapper extends BaseMapper<weibo> {
    @Select("${WeiboPositive}")
    String SqlString(@Param("WeiboPositive")String sql);

    @Select("${findtext}")
    List<weiboETO> FindText(@Param("findtext")String sql);

}
